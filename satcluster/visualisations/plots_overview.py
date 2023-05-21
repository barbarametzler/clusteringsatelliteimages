### Jan 2023
## A. Barbara Metzler


## import packages
import argparse
import numpy as np
import pandas as pd

#import libpysal
import geopandas as gp
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, QuantileTransformer


import matplotlib.patches as mpatches
from matplotlib.cm import get_cmap
import matplotlib.colors as colors_plt
from matplotlib.colors import ListedColormap
from matplotlib import gridspec

import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Create visualisations')

    parser.add_argument('--city', '-c', type=str, default='', help='city')
    parser.add_argument('--gdf', '-df', type=str, default='', help='path to file')
    #parser.add_argument('--list_features', '-lf', type=str, default=['area_sum', 'area_mean', 'bui_count', 'orientation_mean',
    #'all_length', 'major_length','dist_allroads', 'dist_mroads', 'pop_mean', 'ndvi_mean'], help='list of features')
    parser.add_argument('--model', '--m', type=str, default='lr01',
                        help='Model results to plot')
    parser.add_argument('--output_folder', '-of', type=str, default='/home/bmetzler/Documents', help='folder where visualisations are saved')

    return parser.parse_args()


def prep_df(gdf, model):
    gdf = gp.read_file(gdf)
    list_features = ['area_mean',
       'avg_bui', 'bui_count', 'orientation_mean', 'all_length',
       'major_length', 'dist_allroads', 'dist_mroads', 'pop_mean', 'ndvi_mean']
    
    df = gdf[list_features]
    df = df.fillna({'Name':'.', 'City':'.'}).fillna(0)

    
    df['cluster'] = gdf[model]

    X = df.copy()
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # Since the fit_transform() strips the column headers
    # we add them after the transformation
    X_std = pd.DataFrame(X_std, columns=X.columns)

    X_mean = pd.concat([pd.DataFrame(X.mean().drop('cluster'), columns=['mean']), 
                       X.groupby('cluster').mean().T], axis=1)

    X_dev_rel = X_mean.apply(lambda x: round((x-x['mean'])/x['mean'],2)*100, axis = 1)
    X_dev_rel.drop(columns=['mean'], inplace=True)
    X_mean.drop(columns=['mean'], inplace=True)

    X_std_mean = pd.concat([pd.DataFrame(X_std.mean().drop('cluster'), columns=['mean']), 
                       X_std.groupby('cluster').mean().T], axis=1)

    X_std_dev_rel = X_std_mean.apply(lambda x: round((x-x['mean'])/x['mean'],2)*100, axis = 1)
    X_std_dev_rel.drop(columns=['mean'], inplace=True)
    X_std_mean.drop(columns=['mean'], inplace=True)

    return df, X_dev_rel, X_std_dev_rel, X_std_mean


def feature_dists(df, output_folder, city, model):
    colors = ['#9EBD6E','#81a094','#775b59','#32161f', '#946846', '#E3C16F', '#fe938c', '#E6B89C','#EAD2AC',
          '#DE9E36', '#4281A4','#37323E','#95818D'
          ]

    list_features = ['area_mean',
       'avg_bui', 'bui_count', 'orientation_mean', 'all_length',
       'major_length', 'dist_allroads', 'dist_mroads', 'pop_mean', 'ndvi_mean']
    ncols = 4
    nrows = len(list_features) // ncols + (len(list_features) % ncols > 0)
    fig = plt.figure(figsize=(15,15))

    X = df.copy()

    for n, list_features in enumerate(list_features):
        ax = plt.subplot(nrows, ncols, n + 1)
        box = X[[list_features, 'cluster']].boxplot(by='cluster',ax=ax,return_type='both',patch_artist = True)

        for row_key, (ax,row) in box.iteritems():
            ax.set_xlabel('cluster')
            ax.set_title(list_features,fontweight="bold")
            for i,box in enumerate(row['boxes']):
                box.set_facecolor(colors[i])

    fig.suptitle('Feature distributions per cluster', fontsize=18, y=1)   
    plt.tight_layout()
    plt.savefig(str(output_folder)+'/'+str(city) +'/_feature_distributions_' + str(model) + '.png')


def cluster_comparison_bar(df, X_comparison, colors, output_folder, city, model, deviation=True):
    X = df.copy()
    colors = ['#9EBD6E','#81a094','#775b59','#32161f', '#946846', '#E3C16F', '#fe938c', '#E6B89C','#EAD2AC',
          '#DE9E36', '#4281A4','#37323E','#95818D'
          ]

    features = X_comparison.index
    ncols = 3
    # calculate number of rows
    nrows = len(features) // ncols + (len(features) % ncols > 0)
    # set figure size
    fig = plt.figure(figsize=(15,15), dpi=200)
    #interate through every feature
    for n, feature in enumerate(features):
        # create chart
        ax = plt.subplot(nrows, ncols, n + 1)
        X_comparison[X_comparison.index==feature].plot(kind='bar', ax=ax, title=feature, 
                                                             color=colors[0:X.cluster.nunique()],
                                                             legend=False
                                                            )
        plt.axhline(y=0)
        x_axis = ax.axes.get_xaxis()
        x_axis.set_visible(False)

    c_labels = X_comparison.columns.to_list()
    c_colors = colors[0:3]
    mpats = [mpatches.Patch(color=c, label=l) for c,l in list(zip(colors[0:X.cluster.nunique()],
                                                                  X_comparison.columns.to_list()))]

    fig.legend(handles=mpats,
               ncol=ncols,
               loc="upper center",
               fancybox=True,
               bbox_to_anchor=(0.5, 0.98)
              )
    axes = fig.get_axes()
    
    #fig.suptitle(title, fontsize=18, y=1)
    #fig.supylabel('Deviation from overall mean in %')
    plt.ylabel('Deviation from overall mean in %')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(str(output_folder) + '/' + str(city) + '/_cluster_comparison_barplots_'+str(model)+'.png')

def cluster_comparison_one(df, X_dev_rel, colors, output_folder, city, model):
    
    colors = ['#9EBD6E','#81a094','#775b59','#32161f', '#946846', '#E3C16F', '#fe938c', '#E6B89C','#EAD2AC',
          '#DE9E36', '#4281A4','#37323E','#95818D'
          ]
    X = df.copy()

    fig = plt.figure(figsize=(10,5), dpi=200)
    X_dev_rel.T.plot(kind='bar', 
                           ax=fig.add_subplot(), 
                           title="Cluster characteristics", 
                           color=colors,
                           xlabel="Cluster",
                           ylabel="Deviation from overall mean in %"
                          )
    plt.axhline(y=0, linewidth=1, ls='--', color='black')
    plt.legend(bbox_to_anchor=(1.04,1))
    fig.autofmt_xdate(rotation=0)
    plt.tight_layout()
    plt.savefig(str(output_folder) + '/' + str(city) + '/_cluster_comparison_one_'+str(model)+'.png')
    
    
def df_quantile_plot(gdf, X_dev_rel, colors, output_folder, city, model):
    colors = ['#312B33', '#827487', '#D0CBD2', '#88549F', '#F0C922', '#D2B01E','#D2951E','#9E7016', '#fe938c', '#9EBD6E']      
    #color_4 = 'green'
    gray_color = 'gray'

    #colors=['darkgray','gray','dimgray','lightgray']

    list_features = ['area_mean', 'bui_count',  'avg_bui', 'orientation_mean',
    'all_length', 'major_length','dist_allroads', 'dist_mroads', 'pop_mean', 'ndvi_mean']
    
    gdf = gp.read_file(gdf)
    df = gdf[list_features]
    
    df.fillna({'area_mean': 0, 'avg_bui':0, 'bui_count':0,
    'pop_mean':0, 'ndvi_mean':0}, inplace=True) # 'all_length':0, 'major_length':0,

    df = df.rename({'area_mean': 'B. area', 'avg_bui': 'Avg. b. size', 'bui_count':'B. count', 'orientation_mean': 'B. orient.',
    'ndvi_mean': 'NDVI', 'pop_mean': 'Pop. den.', 'dist_allroads': 'Dist. all roads', 'dist_mroads': 'Dist. m. roads',
    'all_length': 'All roads', 'major_length': 'M. roads'}, axis='columns')

    X = df.copy()
    scaler = QuantileTransformer()
    X_quant = scaler.fit_transform(X)

    # Since the fit_transform() strips the column headers
    # we add them after the transformation

    X_quant = pd.DataFrame(X_quant, columns=X.columns)
    X_quant['cluster'] = gdf[model]

    d = X_quant.groupby('cluster').agg('median')
    sel = d.copy()
    sel = sel.sub(0.5)   
    sel = sel.fillna(0)
    
    #colors = [color_4[i] if i == cl_number else gray_color for i in sel.index]
    #print(d)

    fig, ax = plt.subplots(1, figsize=(10,5), dpi=200)
    bar = sel.plot(kind='bar', 
                           ax=ax, 
                           #title="Cluster characteristics", 
                           color=colors,
                           xlabel="Cluster",
                           ylabel="Quantile",
                           legend=False
                          )
    plt.axhline(y=0, linewidth=1, ls='--', color='black')
    #if ticks == True:
    
    #plt.legend(bbox_to_anchor=(1.04,1))
    #fig.autofmt_xdate(rotation=0)
    #plt.tight_layout()
    
    ax.set_yticks([-0.5, -0.25, 0, 0.25, 0.5], [0, 0.25, 0.5, 0.75, 1]) #, rotation='vertical')
    #plt.box(True)
    #if ticks == False:
    #    axis.set_yticks([])
    plt.legend(bbox_to_anchor=(1.22,1)) #bbox_to_anchor=(1.04,1)
    fig.autofmt_xdate(rotation=0)
    #ax.spines['top'].set_visible(False)
    #ax.spines['right'].set_visible(False)
    #ax.spines['bottom'].set_visible(False)
    #ax.spines['left'].set_visible(False)
    
    plt.tight_layout()
    
    plt.savefig(str(output_folder) + '/' + str(city) + '/_cluster_comparison_quantile_'+str(model)+'.png')

def main(args): #str(args.features)
    
    colors = ['#9EBD6E','#81a094','#775b59','#32161f', '#946846', '#E3C16F', '#fe938c', '#E6B89C','#EAD2AC',
          '#DE9E36', '#4281A4','#37323E','#95818D'
          ]

    ## prep data
    df, X_dev_rel, X_std_dev_rel, X_std_mean = prep_df(str(args.gdf), str(args.model))
    
    ## create visualisation
    feature_dists(df,str(args.output_folder), str(args.city), str(args.model))
    print('made feature dists plot')
    
    cluster_comparison_bar(df, X_dev_rel, colors, str(args.output_folder), str(args.city), str(args.model))
    print('made cluster comparison bar plot')

    cluster_comparison_one(df, X_dev_rel, colors, str(args.output_folder), str(args.city), str(args.model))
    print('made cluster comparison in one plot')

    df_quantile_plot(str(args.gdf), X_dev_rel, colors, str(args.output_folder), str(args.city), str(args.model))
    print('made cluster comparison in one plot')

if __name__ == '__main__':
    args = parse_args()
    main(args)
