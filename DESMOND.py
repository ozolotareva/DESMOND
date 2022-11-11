import argparse
import os
import sys

from method import run_DESMOND

parser = argparse.ArgumentParser(
    description="""Searches for gene sets differentially expressed in an unknown subgroup of samples and connected in the network.""", formatter_class=argparse.RawTextHelpFormatter)


parser.add_argument('-e', '--exprs', dest='exprs_file', type=str,
                    help='Expression matrix with sample name in columns and gene in rows, tab-separated.', default='', required=True, metavar='exprs_zscores.tsv')
parser.add_argument('-n', '--network', dest='network_file', type=str,
                    help='Network .tsv file, first two columns contain connected nodes.', default='', required=True, metavar='network.tsv')
parser.add_argument('-d', '--direction', dest='direction', type=str,
                    help='Direction of dysregulation: UP or DOWN',
                    default='UP', required=False)
parser.add_argument('-basename', '--basename', dest='basename', type=str,
                    help='Output basename without extention. If no outfile name provided output will be set "results_hh:mm_dddd-mm-yy".', default='', metavar="results", required=False)
parser.add_argument('-o', '--out_dir', dest='out_dir', type=str,
                    help='Output directory.',
                    default=os.getcwd(), required=False, metavar="result_dir/")
# sampling parameters ###
parser.add_argument('--alpha', dest='alpha', type=float,
                    help='Alpha.', default=0.5, required=False, metavar=0.5)
parser.add_argument('--beta_K', dest='beta_K', type=float,
                    help='Beta/K.', default=1.0, required=False, metavar=1.0)
parser.add_argument('--p_val', dest='p_val', type=float,
                    help='Significance threshold for RRHO method.',
                    default=0.01, required=False, metavar=0.01)
parser.add_argument('--max_n_steps', dest='max_n_steps', type=int,
                    help='Maximal number of steps.', default=200,
                    required=False, metavar=200)
parser.add_argument('--n_steps_averaged', dest='n_steps_averaged', type=int,
                    help='Number of last steps analyzed when checking convergence condition. Values less than 10 are not recommended.', default=20, required=False, metavar=20)
parser.add_argument('--n_steps_for_convergence',
                    dest='n_steps_for_convergence', type=int,
                    help='Required number of steps when convergence conditions is satisfied.', default=5, required=False, metavar=5)
parser.add_argument('-ns', '--min_n_samples', dest='min_n_samples', type=int,
                    help='Minimal number of samples on edge. If not specified, set to max(10,0.1*cohort_size).', default=-1, required=False, metavar="max(10,0.1*cohort_size)")
# merging and filtering parameters
parser.add_argument('-q', '--SNR_quantile', dest='q', type=float,
                    help='Quantile determining minimal SNR threshold.',
                    default=0.1, required=False, metavar=0.1)

# plot flag
parser.add_argument('--plot_all', dest='plot_all', action='store_true',
                    help='Switches on all plotting.', required=False)
parser.add_argument('--report_merging', dest='report_merging',
                    action='store_true',
                    help='Report all merging attempts (for debugging).',
                    required=False)
parser.add_argument('--force', dest='force', action='store_true',
                    help='Overwrite precomputed intermediate results.',
                    required=False)
# if verbose
parser.add_argument('--verbose', dest='verbose',
                    action='store_true', help='', required=False)


def main(args):

    # where to write the results ####
    # create directory if it does not exists
    if not os.path.exists(args.out_dir) and not args.out_dir == ".":
        os.makedirs(args.out_dir)
    args.out_dir = args.out_dir + "/"

    # Merge remaining biclusters
    run_DESMOND(args.exprs_file, args.network_file,
                direction=args.direction, min_n_samples=args.min_n_samples,
                p_val=args.p_val, alpha=args.alpha, beta_K=args.beta_K,
                out_dir=args.out_dir, basename=args.basename, q=args.q,
                max_n_steps=args.max_n_steps,
                n_steps_averaged=args.n_steps_averaged,
                n_steps_for_convergence=args.n_steps_for_convergence,
                force=args.force, plot_all=args.plot_all,
                report_merging=args.report_merging, verbose=args.verbose)

    return True


if __name__ == '__main__':

    # Step 1. Read and check inputs ####

    # Disable during debugging
    args = parser.parse_args()

    # ! Disable during run
    # UP and DOWN
    # args = argparse.Namespace(
    #     exprs_file="/home/fabio/Downloads/desmod_run/D1_test/test_df.tsv",
    #     network_file='/home/fabio/Downloads/unpast_trans/data/bicon_network.tsv',
    #     basename='TEST',
    #     out_dir='/home/fabio/Downloads/desmod_run/D1_test/TEST',
    #     alpha=0.5,
    #     p_val=0.01,
    #     q=0.5,
    #     # direction='UP',
    #     direction='DOWN',
    #     beta_K=1.0,
    #     max_n_steps=200,
    #     n_steps_averaged=20,
    #     n_steps_for_convergence=5,
    #     min_n_samples=-1,
    #     force=False,
    #     plot_all=False,
    #     report_merging=False,
    #     verbose=True)

    if args.verbose:
        print("Expression:", args.exprs_file,
              "\nNetwork:", args.network_file,
              "\n", file=sys.stdout)
        print("\nRRHO significance threshold:", args.p_val,
              "\nSNR_quantile:", args.q,
              "\nalpha:", args.alpha,
              "\nbeta/K:", args.beta_K,
              "\ndirection:", args.direction,
              "\nmax_n_steps:", args.max_n_steps,
              "\nn_steps_averaged:", args.n_steps_averaged,
              "\nn_steps_for_convergence:", args.n_steps_for_convergence,
              "\n", file=sys.stdout)
    main(args)
