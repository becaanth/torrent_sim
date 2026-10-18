from plot_progress import *

EXPERIMENT = 'convoy' # or 'multi'
# EXPERIMENT = 'multi'
if __name__ == '__main__':
    results_dirs = [f'results/checkpoint/sweep_{policy}_base_{EXPERIMENT}' for policy in ['s','rr','c']]

    # for key, item in params.items():
    #     print(key, item)
    #     fig = plot_progress(
    #         sweep_dir=item[0],
    #         color_by="agents.peers.0.strategy",
    #         tint_by=item[1],
    #         tint_values=item[2],
    #         spread='peers',
    #         split_chain=True,
    #         out=f'plots/{EXPERIMENT}/sweep_{key}.pdf'
    #     )
    # 1x3 header figure: one panel per sweep, panel order = order of `params`
    if EXPERIMENT == 'convoy':
        params = {
            'rarest-random' : (results_dirs[1], 'radio.fixed.d0', (200,300,400,500,1000,2000)), 
            'cascading' : (results_dirs[2], 'heterogeneity.slow_factor', (0.01,0.5,0.75,0.9,0.99)),
            'sequential' : (results_dirs[0], 'agents.peers.0.count', (2,3,4,5,10,15,20,40))
        }
        plot_progress_row(
            [dict(sweep_dir=d, tint_by=k, tint_values=v) for d, k, v in params.values()],
            out=f'plots/{EXPERIMENT}/progress_row.pdf',
            spread='peers',
            split_chain=True,
        )
    else:
        params = {
            'rarest-random' : (results_dirs[1], 'radio.fixed.d0', (200,500,1000)), 
            'cascading' : (results_dirs[2], 'heterogeneity.slow_factor', (0.01,0.5,0.99)),
            'sequential' : (results_dirs[0], 'agents.peers.0.count', (4,10,20,40))
        }
        plot_progress_row(
            [dict(sweep_dir=d, tint_by=k, tint_values=v) for d, k, v in params.values()],
            out=f'plots/{EXPERIMENT}/progress_row.pdf',
            split_chain=True,
        )