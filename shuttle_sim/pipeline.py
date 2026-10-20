"""Simulation runner: actual electrode voltages throughout, no virtual score API."""
from dataclasses import asdict
from pathlib import Path
import csv, hashlib, json, platform, sys, time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from .field import finite, Rejected, load_trap
from .hardware import Hardware
from .dynamics import propagate, validate_full_field_centroid


def write_json(path,value):
    def conv(x):
        if isinstance(x,np.ndarray): return x.tolist()
        if isinstance(x,np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=conv,allow_nan=False),encoding='utf8')
    tmp.replace(path)


def save_csv(path,header,array):
    with open(path,'w',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(array)


class Runner:
    def __init__(self,config):
        self.cfg=config
        self.trap=load_trap(config['trap'])
        self.hw=Hardware(**config['hardware'])
        self.hw.validate(len(self.trap.indices))
        self.task=config['task']
        if config.get('robust_samples',8)<1:raise ValueError('positive robust sample count required')
        finite([self.task['min_frequency_hz'],self.task['distance_m'],self.task['duration_s'],self.task['hold_s'],
                self.task['max_n_total'],self.task['position_tolerance_m']], 'task')
        if self.task['min_frequency_hz']<=0 or self.task['duration_s']<=0 or self.task['hold_s']<0 or self.task['max_n_total']<=0 or self.task['position_tolerance_m']<=0:
            raise ValueError('invalid task scales')
        if self.task['protocol'] not in ['one_way','round_trip','repeated']:
            raise ValueError('unknown protocol')
        if int(self.task.get('cycles',1))<1:
            raise ValueError('positive cycles required')
        self.initial_V=finite(config['trap']['initial_dc_V'],'initial voltages')
        if self.initial_V.shape!=self.trap.lo.shape or np.any(self.initial_V<self.trap.lo) or np.any(self.initial_V>self.trap.hi):
            raise ValueError('initial voltages out of bounds')
        self.initial=self.trap.local(self.initial_V,config['trap']['minimum_seed_m'])
        self.direction=finite(self.task['direction'],'direction')
        if self.direction.shape!=(3,) or np.linalg.norm(self.direction)==0: raise ValueError('nonzero 3D direction required')
        self.direction/=np.linalg.norm(self.direction)
        self.start=self.initial.position_m
        self.target=self.start+self.task['distance_m']*self.direction
        self.failures=[];self.history=[];self.checkpoint=None

    def prepare(self):
        """Geometry-dependent support solutions with hard force constraints."""
        count=int(self.cfg.get('route_nodes',41))
        if count<9: raise ValueError('at least 9 route nodes required')
        self.sroute=np.linspace(-.05,1.05,count)
        vs=[];ls=[]
        prev=self.initial_V.copy()
        for s in self.sroute:
            p=self.start+s*self.task['distance_m']*self.direction
            v,loc=self.trap.invert(p,self.direction,self.initial.omega_rad_s[0],prev)
            vs.append(v);ls.append(loc);prev=v
        self.route_V=np.array(vs)
        self.voltage_spline=CubicSpline(self.sroute,self.route_V,axis=0)
        # Linear-response ROM anchored on geometry, NEVER a hand-written parabola.
        self.route_r=np.array([z.position_m for z in ls])
        self.route_H=np.array([z.hessian_J_m2 for z in ls])
        self.route_G=np.array([self.trap.basis(r,1) for r in self.route_r])
        self.route_BH=np.array([self.trap.basis(r,2) for r in self.route_r])
        self.Hspline=CubicSpline(self.sroute,self.route_H,axis=0)
        self.Gspline=CubicSpline(self.sroute,self.route_G,axis=0)
        self.BHspline=CubicSpline(self.sroute,self.route_BH,axis=0)
        _,_,R=np.linalg.svd(self.route_V-self.route_V.mean(axis=0),full_matrices=False)
        self.correction_modes=R[:2]
        self.support_diagnostics=[]
        from scipy.optimize import linear_sum_assignment
        last=None
        labels=np.arange(3)
        for s,z,v in zip(self.sroute,ls,self.route_V):
            overlaps=None if last is None else np.abs(last.T@z.modes)
            if overlaps is not None:
                _,cols=linear_sum_assignment(-overlaps**2)
                mapped=labels.copy();mapped[cols]=labels;labels=mapped
            self.support_diagnostics.append({'s':s,'position_m':z.position_m,'frequencies_hz':z.omega_rad_s/(2*np.pi),
                'modes_columns':z.modes,'voltage_V':v,'adjacent_mode_overlap':overlaps,
                'tracked_labels_for_sorted_modes':labels.copy(),
                'mode_rotation_risk':bool(overlaps is not None and np.min(np.max(overlaps,axis=1))<.95),
                'relative_min_gap':float(np.min(np.diff(z.omega_rad_s))/z.omega_rad_s.max())})
            last=z.modes

    def grid(self):
        fs=self.hw.sample_rate_hz
        T0=self.task['duration_s']; hold=self.task['hold_s']
        nleg=round(T0*fs);nhold=round(hold*fs)
        if nleg<20 or abs(nleg/fs-T0)>1e-12 or abs(nhold/fs-hold)>1e-12:
            raise ValueError('duration/hold must be integer DAC periods; >=20 leg samples')
        protocol=self.task['protocol'];cycles=self.task.get('cycles',1) if protocol=='repeated' else 1
        legs=1 if protocol=='one_way' else 2*cycles
        chunks=[]
        for leg in range(legs):
            s=np.linspace(0,1,nleg+1)
            direction=1 if leg%2==0 else -1
            # Every endpoint appears exactly once. Holds are explicit, filter
            # states and dynamics persist through turns and repeated cycles.
            chunks.append((s[:-1],direction,False))
            if nhold: chunks.append((np.ones(nhold),direction,True))
        s,d,h=zip(*[(x,di,ho) for a,di,ho in chunks for x in a])
        s=np.r_[s,1.];d=np.r_[d,1 if (legs-1)%2==0 else -1];h=np.r_[h,True]
        return np.arange(len(s))/fs,s,np.array(d),np.array(h),nleg/fs

    def commands(self,theta,baseline='smooth'):
        theta=finite(theta,'theta')
        if theta.shape!=(8,): raise ValueError('8 corrective parameters required')
        t,s,d,holds,T0=self.grid()
        # Local lead vanishes with value and slope at leg boundaries.
        from numpy.polynomial import Polynomial as P
        env=P([0,0,16,-32,16]);x=P([0,1])
        shape=x+theta[0]*env+theta[1]*env*P([-1,2])
        ss=shape(s)
        if ss.min() < -1e-10 or ss.max() > 1+1e-10:
            raise Rejected('route_corridor','nonphysical time-law overshoot')
        envelope=env(s)
        if baseline=='smooth':
            poly=P([0,0,0,10,-15,6])(shape)
            q=poly(s)
        elif baseline=='sta':
            poly=P([0,0,0,0,35,-84,70,-20])(shape)
            q=poly(s)+poly.deriv(2)(s)/(T0*self.initial.omega_rad_s[0])**2
        else: raise ValueError('baseline must be smooth or sta')
        route=np.where(d>0,q,1-q)
        if route.min()<self.sroute[0] or route.max()>self.sroute[-1]:
            raise Rejected('route_corridor','reference outside supported route')
        v=self.voltage_spline(route)
        basis=np.column_stack([envelope,envelope*(2*s-1),envelope*(6*s*s-6*s+1)])
        correction=(basis@theta[2:].reshape(3,2))@self.correction_modes
        v+=correction
        if np.any(v<self.trap.lo) or np.any(v>self.trap.hi):
            raise Rejected('infeasible_voltages','command outside electrode bounds')
        return t,v,route

    def rom(self,t,delivered,command_times,route,stray=None,exact_nodes=None):
        sr=np.interp(t,command_times,route)
        r=self.start+sr[:,None]*self.task['distance_m']*self.direction
        if exact_nodes:
            idx=np.unique(np.r_[np.linspace(0,len(t)-1,exact_nodes).astype(int),len(t)-1])
            locals=[]
            for i in idx:
                z=self.trap.local(delivered[i],r[i],stray);locals.append(z)
            rr=np.array([z.position_m for z in locals]);hh=np.array([z.hessian_J_m2 for z in locals])
            r=CubicSpline(t[idx],rr,axis=0)(t);H=CubicSpline(t[idx],hh,axis=0)(t)
            final=locals[-1]
            diag=[{'time_s':t[i],'position_m':z.position_m,'omega_rad_s':z.omega_rad_s,
                   'modes_columns':z.modes} for i,z in zip(idx,locals)]
        else:
            dv=delivered-self.voltage_spline(sr)
            g=np.einsum('nc,nci->ni',dv,self.Gspline(sr))
            H=self.Hspline(sr)+np.einsum('nc,ncij->nij',dv,self.BHspline(sr))
            if stray is not None: g-=self.trap.q*np.asarray(stray)
            r=r-np.linalg.solve(H,g[...,None])[...,0]
            final=self.trap.local(delivered[-1],r[-1],stray);diag=[]
        eig=np.linalg.eigvalsh(H/self.trap.mass)
        if eig.min()<(2*np.pi*self.task['min_frequency_hz'])**2:
            raise Rejected('weak_confinement','frequency below task floor')
        nominal=self.start+sr[:,None]*self.task['distance_m']*self.direction
        max_shift=float(np.max(np.linalg.norm(r-nominal,axis=1)))
        if max_shift>self.cfg.get('rom_max_shift_m',20e-6):
            raise Rejected('rom_trust_region','delivered minimum outside validated local corridor')
        return r,H,final,diag

    def evaluate(self,theta,baseline='smooth',ideal=False,hardware=None,stray=None,
                 exact_nodes=None,method='exact',subdivisions=None,keep=False,frequency_scale=1.):
        ct,cmd,route=self.commands(theta,baseline)
        hw=hardware or self.hw
        t,v,codes,dist=hw.deliver(cmd,subdivisions or self.cfg.get('subdivisions',4),ideal)
        if np.any(v<self.trap.lo) or np.any(v>self.trap.hi):
            raise Rejected('hardware_saturation','delivered voltage outside trap bounds')
        if ideal:
            # No interpolation across DAC jumps. One state per actual command.
            t=ct;v=cmd.copy()
            if exact_nodes:
                locals=[self.trap.local(vv,self.start+ss*self.task['distance_m']*self.direction,stray) for vv,ss in zip(v,route)]
                r=np.array([z.position_m for z in locals]);H=np.array([z.hessian_J_m2 for z in locals])
                final=locals[-1];diag=[]
            else:r,H,final,diag=self.rom(t,v,ct,route,stray,None)
        else:r,H,final,diag=self.rom(t,v,ct,route,stray,exact_nodes)
        init=self.trap.local(v[0],self.start,stray)
        if frequency_scale!=1.:
            from .field import Local
            if not np.isfinite(frequency_scale) or frequency_scale<=0: raise ValueError('invalid frequency scale')
            H=H*frequency_scale**2
            init=Local(init.position_m,init.hessian_J_m2*frequency_scale**2,init.omega_rad_s*frequency_scale,init.modes)
            final=Local(final.position_m,final.hessian_J_m2*frequency_scale**2,final.omega_rad_s*frequency_scale,final.modes)
        target=self.target if self.task['protocol']=='one_way' else self.start
        result=propagate(t,r,H,init,final,self.trap.mass,target,method,hold='zero_order' if ideal else 'linear')
        result['distortion']=dist
        result['accepted']=bool(result['n_total']<=self.task['max_n_total'] and
             result['position_error_m']<=self.task['position_tolerance_m'] and
             result['well_target_error_m']<=self.task['position_tolerance_m'])
        result['quality_failures']=[]
        if result['n_total']>self.task['max_n_total']:result['quality_failures'].append('excessive_excitation')
        if result['position_error_m']>self.task['position_tolerance_m']:result['quality_failures'].append('transport_accuracy')
        if result['well_target_error_m']>self.task['position_tolerance_m']:result['quality_failures'].append('well_target_accuracy')
        result['model']='3D geometry-derived local harmonic Gaussian'
        result['minimum_frequency_hz']=float(np.sqrt(np.linalg.eigvalsh(H/self.trap.mass).min())/(2*np.pi))
        if keep:
            return result,dict(t=t,v=v,codes=codes,ct=ct,cmd=cmd,route=route,r=r,H=H,initial=init,final=final,diag=diag)
        return result

    def scenarios(self,count,seed):
        spec=self.cfg['uncertainty'];rng=np.random.default_rng(seed);C=len(self.trap.indices)
        rows=[]
        for i in range(count):
            h=asdict(self.hw)
            h['gain']=(np.broadcast_to(self.hw.gain,(C,))*(1+rng.normal(0,spec['gain_sigma'],C))).tolist()
            h['offset_V']=(np.broadcast_to(self.hw.offset_V,(C,))+rng.normal(0,spec['offset_sigma_V'],C)).tolist()
            h['delay_s']=(np.broadcast_to(self.hw.delay_s,(C,))+rng.uniform(0,spec['additional_delay_max_s'],C)).tolist()
            h['lowpass_hz']=(np.asarray(h['lowpass_hz'])*np.exp(rng.normal(0,spec['log_filter_sigma']))).tolist()
            rows.append({'scenario':i,'hardware':h,'stray_V_m':rng.normal(0,spec['stray_sigma_V_m'],3).tolist(),
                         'frequency_scale':float(np.exp(rng.normal(0,spec.get('log_frequency_sigma',0.))))})
        return rows

    def optimize(self,ideal=False,robust=False):
        history=[];best=[float('inf'),np.zeros(8)]
        training=self.scenarios(self.cfg.get('robust_training_samples',3),self.cfg['uncertainty']['seed']+1) if robust else []
        def loss(x):
            try:
                results=[self.evaluate(x,baseline='sta',ideal=ideal)]
                for sc in training:
                    results.append(self.evaluate(x,'sta',hardware=Hardware(**sc['hardware']),stray=sc['stray_V_m'],
                                                 frequency_scale=sc['frequency_scale']))
                vals=[]
                for r in results:
                    val=np.log1p(r['n_total']/self.task['max_n_total'])
                    val+=np.log1p((r['position_error_m']/self.task['position_tolerance_m'])**2)
                    val+=np.log1p((r['well_target_error_m']/self.task['position_tolerance_m'])**2)
                    vals.append(val)
                value=float(.35*np.mean(vals)+.65*np.max(vals)+1e-4*np.sum(x*x))
                status='ok'
            except Rejected as ex:
                value=1e6;status=ex.code
            except (np.linalg.LinAlgError,FloatingPointError) as ex:
                value=1e6;status='numerical_failure'
            if value<best[0]:best[:]=[float(value),x.copy()]
            history.append({'evaluation':len(history),'loss':float(value),'theta':x.tolist(),'status':status})
            if self.checkpoint is not None:
                write_json(self.checkpoint,{'best_loss':best[0],'theta':best[1],'history':history,'complete':False})
            return value
        bounds=[(-.10,.10),(-.10,.10)]+[(-.4,.4)]*6
        # Hard function budget. Preserve an evaluated baseline even if optimizer
        # stops before a complete direction-set sweep.
        loss(np.zeros(8))
        opt=minimize(loss,best[1],method='Powell',bounds=bounds,
                     options={'maxfev':self.cfg.get('optimization_evaluations',80),'xtol':1e-3,'ftol':1e-4})
        out={'theta':best[1].tolist(),'loss':best[0],'optimizer_success':bool(opt.success),
             'termination':str(opt.message),'evaluations':len(history),'history':history,
             'failure_mode':None if opt.success else 'optimizer_budget_exhaustion_or_stagnation',
             'parameterization':'2 time-law coefficients + 6 voltage corrections (2 geometry SVD modes)',
             'search_bounds':bounds,'ideal_objective':ideal,'robust_training_scenarios':training}
        if self.checkpoint is not None:write_json(self.checkpoint,{**out,'complete':True})
        return out

    def robust(self,theta,count=12):
        """Held-out, declared Monte Carlo simulation. No experimental claims."""
        spec=self.cfg['uncertainty'];rows=[]
        for sc in self.scenarios(count,spec['seed']):
            row=dict(sc)
            try:
                row.update(self.evaluate(theta,'sta',hardware=Hardware(**sc['hardware']),stray=sc['stray_V_m'],
                    frequency_scale=sc['frequency_scale'],exact_nodes=641,subdivisions=8))
                row['status']='ok'
            except Rejected as ex:row.update(status=ex.code,accepted=False)
            rows.append(row)
        values=np.array([r['n_total'] for r in rows if r['status']=='ok'])
        valid=len(values)==count
        return {'assumption':'unmeasured uncertainty distribution; simulation only',
                'all_scenarios_valid':valid,'pass_fraction':sum(r['accepted'] for r in rows)/count,
                'mean_n':float(values.mean()) if len(values) else None,
                'worst_n':float(values.max()) if valid else None,
                'cvar_20pct_n':float(np.sort(values)[-max(1,int(np.ceil(.2*count))):].mean()) if valid else None,
                'rows':rows,'seed':spec['seed']}

    def run(self,out,full_field=False):
        out=Path(out);out.mkdir(parents=True,exist_ok=True);self.checkpoint=out/'optimizer_checkpoint.json'
        started=time.time();self.prepare();write_json(out/'config.json',self.cfg)
        write_json(out/'route_diagnostics.json',self.support_diagnostics)
        comparison={}
        for name,kind in [('smooth','smooth'),('sta','sta')]:
            comparison[name+'_ideal']=self.evaluate(np.zeros(8),kind,ideal=True,exact_nodes=321,subdivisions=8)
            comparison[name+'_delivered']=self.evaluate(np.zeros(8),kind,exact_nodes=321,subdivisions=8)
        self.checkpoint=out/'unaware_optimizer_checkpoint.json';unaware=self.optimize(ideal=True)
        write_json(out/'unaware_optimization.json',unaware)
        comparison['hardware_unaware_optimized_delivered']=self.evaluate(unaware['theta'],'sta',exact_nodes=321,subdivisions=8)
        self.checkpoint=out/'aware_optimizer_checkpoint.json';aware=self.optimize(robust=True)
        write_json(out/'aware_optimization.json',aware)
        candidates=[('sta',np.zeros(8)),('unaware',np.array(unaware['theta'])),('aware',np.array(aware['theta']))]
        # Rerank under actual local-minimum reconstruction, not the search ROM.
        rerank=[]
        for name,theta in candidates:
            r=self.evaluate(theta,'sta',exact_nodes=321,subdivisions=8)
            score=r['n_total']+1e8*r['well_target_error_m']+1e8*r['position_error_m']
            rerank.append((score,name,theta,r))
        _,selected,theta,r=sorted(rerank,key=lambda z:z[0])[0]
        write_json(out/'candidate_reranking.json',[{'source':n,'score':sc,'theta':th,'metrics':rr} for sc,n,th,rr in rerank])
        comparison['hardware_aware_candidate_delivered']=rerank[-1][3]
        comparison['selected']=r
        write_json(out/'comparison.json',comparison)
        r,trace=self.evaluate(theta,'sta',exact_nodes=641,subdivisions=8,keep=True)
        refined=self.evaluate(theta,'sta',exact_nodes=1281,subdivisions=16)
        ind=self.evaluate(theta,'sta',exact_nodes=641,subdivisions=8,method='dop853')
        cheap=self.evaluate(theta,'sta')
        validation={'cheap_search_model':cheap,'cheap_vs_validated_delta_n':abs(cheap['n_total']-r['n_total']),
                    'midpoint_exact_fine':r,'independent_dop853':ind,'refined_1281_nodes_16_substeps':refined,
                    'refinement_delta_n':abs(r['n_total']-refined['n_total']),
                    'selected_from':selected,'selected_theta':theta,
                    'coarse_to_fine_delta_n':abs(r['n_total']-comparison['selected']['n_total']),
                    'integrator_delta_n':abs(r['n_total']-ind['n_total']),
                    'independence':'Different integration method; same 3D local harmonic physical approximation',
                    'limitation':'No full RF micromotion or non-Gaussian quantum propagation'}
        # Tolerance is explicit and saved, not silently replaced by acceptance.
        tol=self.cfg.get('validation_n_tolerance',.005)
        validation['convergence_passed']=bool(max(validation['refinement_delta_n'],validation['integrator_delta_n'])<=tol)
        validation['n_tolerance']=tol
        if full_field:
            validation['nonlinear_centroid']=validate_full_field_centroid(self.trap,trace['t'],trace['v'],trace['initial'],
                                      self.target if self.task['protocol']=='one_way' else self.start)
        write_json(out/'validation.json',validation)
        hold_scan=[];original_hold=self.task['hold_s']
        try:
            for hold in self.cfg.get('hold_scan_s',[original_hold]):
                self.task['hold_s']=float(hold)
                metric=self.evaluate(theta,'sta',exact_nodes=641,subdivisions=8)
                hold_scan.append({'hold_s':hold,'metrics':metric})
        finally:self.task['hold_s']=original_hold
        write_json(out/'hold_scan.json',hold_scan)
        robust=self.robust(theta,self.cfg.get('robust_samples',8));write_json(out/'robust.json',robust)
        save_csv(out/'ideal_waveform.csv',['time_s']+self.trap.names,np.column_stack([trace['ct'],trace['cmd']]))
        with open(out/'dac_waveform.csv','w',newline='') as f:
            w=csv.writer(f);w.writerow(['time_s']+self.trap.names)
            for ti,codes in zip(trace['ct'],trace['codes']):w.writerow([float(ti)]+[int(code) for code in codes])
        save_csv(out/'delivered_waveform.csv',['time_s']+self.trap.names,np.column_stack([trace['t'],trace['v']]))
        save_csv(out/'delivered_path.csv',['time_s','x_m','y_m','z_m'],np.column_stack([trace['t'],trace['r']]))
        write_json(out/'delivered_local_diagnostics.json',trace['diag'])
        write_json(out/'distortion.json',r['distortion'])
        data=(out/'dac_waveform.csv').read_bytes()
        write_json(out/'waveform_metadata.json',{'version':'1.0','simulation_only':True,
            'execution_authorized':False,'channel_order':self.trap.names,'units':'DAC codes (integer)',
            'sample_period_s':1/self.hw.sample_rate_hz,'code_min_V':self.hw.dac_min_V,
            'code_max_V':self.hw.dac_max_V,'dac_bits':self.hw.dac_bits,
            'hold':'zero_order','last_sample':'terminal command, held indefinitely',
            'clock':'common ideal simulation clock','trigger_latency_s':0,
            'calibration_state':'unmeasured_simulation_assumptions','hardware_model':asdict(self.hw),
            'sha256':hashlib.sha256(data).hexdigest()})
        import scipy
        status={'nominal_quality_passed':bool(r['accepted'] and ind['accepted'] and refined['accepted']),'numerical_validation_passed':validation['convergence_passed'],
                'robust_all_passed':bool(robust['pass_fraction']==1),
                'hold_scan_all_passed':all(z['metrics']['accepted'] for z in hold_scan),
                'experiment_ready':False,'selected':selected,
                'scope':'simulation-only on specified Sion/GDS electrode geometry'}
        status['simulation_acceptance_passed']=bool(status['nominal_quality_passed'] and status['numerical_validation_passed'] and
               status['robust_all_passed'] and status['hold_scan_all_passed'])
        write_json(out/'status.json',status)
        write_json(out/'manifest.json',{'package':'shuttle_sim 0.1.0','python':sys.version,
            'numpy':np.__version__,'scipy':scipy.__version__,'platform':platform.platform(),
            'elapsed_s':time.time()-started,'run_unix_time':started,
            'input_sha256':hashlib.sha256(json.dumps(self.cfg,sort_keys=True).encode()).hexdigest(),
            'source_base_commit':'0485696de5ba90c81be05a088ed920c7096af566',
            'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
            'trap_source':'specified geometry in config; no measured electrode maps',
            'physics':'RF pseudopotential + 3D local harmonic Gaussian',
            'status':status})
        return status
