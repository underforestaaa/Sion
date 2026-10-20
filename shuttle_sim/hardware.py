"""Explicit simulation assumptions, never a claim about measured electronics."""
from dataclasses import dataclass, field
import hashlib, json
import numpy as np
from .field import finite, Rejected


@dataclass
class Hardware:
    sample_rate_hz: float = 20e6
    dac_bits: int = 18
    dac_min_V: float = -20.
    dac_max_V: float = 20.
    lowpass_hz: list = field(default_factory=lambda: [1e6, 500e3])
    gain: object = 1.
    offset_V: object = 0.
    delay_s: object = 0.
    max_jump_V: float = 1.
    identifier: str = 'declared-simulation-chain-v1'

    def validate(self, channels):
        finite([self.sample_rate_hz,self.dac_min_V,self.dac_max_V,self.max_jump_V], 'hardware')
        if self.sample_rate_hz <= 0 or self.dac_min_V >= self.dac_max_V or self.max_jump_V <= 0:
            raise ValueError('invalid hardware scales')
        if not isinstance(self.dac_bits,int) or not 2 <= self.dac_bits <= 32:
            raise ValueError('DAC bits must be integer in [2,32]')
        if np.any(finite(self.lowpass_hz,'poles') <= 0):
            raise ValueError('low-pass poles must be positive')
        for k in ['gain','offset_V','delay_s']:
            np.broadcast_to(finite(getattr(self,k),k),(channels,))
        if np.any(np.asarray(self.delay_s)<0) or np.any(np.asarray(self.gain)<=0):
            raise ValueError('positive gains and nonnegative delays required')

    def deliver(self, commands, subdivisions=4, ideal=False):
        """ZOH at k/fs, exact cascade propagation, DC steady-state prehistory."""
        c = finite(commands,'commands_V')
        if c.ndim != 2 or len(c)<2 or subdivisions < 1 or int(subdivisions)!=subdivisions:
            raise ValueError('commands [samples,channels], positive integer subdivisions required')
        self.validate(c.shape[1])
        clipped = (c<self.dac_min_V)|(c>self.dac_max_V)
        if clipped.any():
            raise Rejected('hardware_saturation','DAC range exceeded')
        jumps = float(np.max(np.abs(np.diff(c,axis=0))))
        if jumps>self.max_jump_V:
            raise Rejected('slew_limit',f'max jump {jumps:g} V')
        lsb = (self.dac_max_V-self.dac_min_V)/(2**self.dac_bits-1)
        codes = np.rint((c-self.dac_min_V)/lsb).astype(np.int64)
        quantized = self.dac_min_V+codes*lsb
        if ideal:
            quantized=c.copy()
        N,C = c.shape
        dt = 1/self.sample_rate_hz/subdivisions
        t = np.arange((N-1)*subdivisions+1)*dt
        gain=np.broadcast_to(self.gain,(C,)); offset=np.broadcast_to(self.offset_V,(C,))
        held=quantized[np.minimum(np.arange(len(t))//subdivisions,N-1)]
        if ideal:
            y=held
        elif not self.lowpass_hz:
            y=held*gain+offset
        else:
            from scipy.linalg import expm
            poles=2*np.pi*np.asarray(self.lowpass_hz)
            P=len(poles); A=-np.diag(poles)
            for j in range(1,P): A[j,j-1]=poles[j]
            B=np.zeros((P,1));B[0]=poles[0]
            M=np.zeros((P+1,P+1));M[:P,:P]=A;M[:P,P:]=B
            E=expm(M*dt);Ad=E[:P,:P];Bd=E[:P,P:]
            state=np.tile((quantized[0]*gain+offset),(P,1));y=np.empty_like(held);y[0]=state[-1]
            for k in range(len(t)-1):
                state=Ad@state+Bd@(held[k]*gain+offset)[None,:]
                y[k+1]=state[-1]
        if not ideal:
            delay=np.broadcast_to(self.delay_s,(C,))
            y=np.column_stack([np.interp(t-delay[j],t,y[:,j],left=y[0,j]) for j in range(C)])
        report={'max_command_jump_V':jumps,'clipping_events':0,'lsb_V':lsb,
                'max_deviation_V':float(np.max(np.abs(y-held))),
                'rms_deviation_V':float(np.sqrt(np.mean((y-held)**2))),
                'channel_max_deviation_V':np.max(np.abs(y-held),axis=0).tolist(),
                'delay_s':np.broadcast_to(self.delay_s,(C,)).tolist(),
                'hold':'zero_order','filter':'exact continuous RC cascade on dense ZOH grid',
                'hardware_identifier':self.identifier,'calibration_state':'unmeasured_simulation_assumptions'}
        return t,y,codes,report


def verify_export(path):
    from pathlib import Path
    p=Path(path);meta=json.loads((p/'waveform_metadata.json').read_text())
    data=(p/'dac_waveform.csv').read_bytes()
    if hashlib.sha256(data).hexdigest()!=meta['sha256']:
        raise Rejected('integrity_failure','waveform checksum mismatch')
    return meta
