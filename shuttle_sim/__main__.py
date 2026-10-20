import argparse, json, sys
from pathlib import Path
from .pipeline import Runner, write_json
from .field import Rejected


def main():
    p=argparse.ArgumentParser(description='Geometry-aware single-ion simulation; NEVER apparatus control')
    p.add_argument('--config',default='configs/shuttle_reference.json')
    p.add_argument('--out',default='output/shuttle_reference')
    p.add_argument('--full-field',action='store_true',help='also validate centroid with nonlinear Sion forces')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(Path(a.config).read_text())
    if cfg['trap']['kind']=='gds':
        cfg['trap']['path']=str((Path(a.config).resolve().parent/cfg['trap']['path']).resolve())
    try:
        status=Runner(cfg).run(out,a.full_field)
    except (Rejected,ValueError,FileNotFoundError) as e:
        write_json(out/'failure.json',{'status':getattr(e,'code','invalid_input'),'message':str(e)})
        print(str(e),file=sys.stderr);return 2
    print(json.dumps(status,indent=2));return 0


if __name__=='__main__':sys.exit(main())
