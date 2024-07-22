import sys
import argparse

from b2heavy.TwoPointFunctions import fitter, types2pts
from b2heavy.TwoPointFunctions import fitter_old

from b2heavy.ThreePointFunctions import types3pts, fitter3pts
from b2heavy.ThreePointFunctions import types3pts
from b2heavy.ThreePointFunctions import corr3pts
from b2heavy.ThreePointFunctions import globalfit


if __name__ == "__main__":
    prs = argparse.ArgumentParser()
    prs.add_argument('-f','--fit'      , action='store_true')
    prs.add_argument('-falt'           , action='store_true')
    prs.add_argument('-t','--type'     , action='store_true')
    prs.add_argument('-r','--ratio'    , action='store_true')
    prs.add_argument('-c3'             , action='store_true')
    prs.add_argument('-ralt'           , action='store_true')
    prs.add_argument('-rf','--ratiofit', action='store_true')
    prs.add_argument('-gf'             , action='store_true')
    prs.add_argument('--do', type=int)
    
    args = prs.parse_args()
    if args.fit:
        # fitter.main(args.do)
        fitter.main()

    elif args.type:
        # types2pts.eff_coeffs(args.do)
        # types2pts.global_eff_coeffs()
        types2pts.main() 

    elif args.ratio:
        types3pts.main()
        # types3pts.main_scan_all_ratio()

    elif args.ratiofit:
        fitter3pts.main()

    elif args.c3:
        corr3pts.main()

    elif args.gf:
        globalfit.main()