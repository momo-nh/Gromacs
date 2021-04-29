#coding=utf-8
import numpy as np 
import pandas as pd 
import mdtraj as md
from pylab import *
import plotly
import plotly.graph_objs as go
import plotly.tools as tls
from plotly.offline import plot as plot_
import os
from math import ceil
import argparse
#from msmbuilder.featurizer import RawPositionsFeaturizer
#from msmbuilder.decomposition import PCA

#from util import *
import warnings 
warnings.filterwarnings(action='ignore')
import matplotlib.pyplot as plt
#import matplotlib.axes.Axes
plt.switch_backend("Agg")
from matplotlib.ticker import MultipleLocator, FormatStrFormatter 



def FEL_pdf_2D(data,bins=50,cmap=None,KbT=-2.479,fig=None,xlabel_='x',ylabel_='y'):
    """

    """
    #data is a 2D array
    if fig == None:
        fig = '2D_pdf_FEL'
    data = np.array(data)
    print(data.shape)
    if len(data.shape) == 3 and data.shape[1] != 2:
        raise ValueError('FEL data should be a or a serise of 2D array-like and shape should be (n,2) or (m,n,2), e.t. [[1,2],[1,3],....] or [[[1,2],[2,2]...],[[2,1],[1,1]...]... ]')
    elif len(data.shape) == 2 and data.shape[0] != 2:
        raise ValueError('FEL data should be a or a serise of 2D array-like and shape should be (n,2) or (m,n,2), e.t. [[1,2],[1,3],....] or [[[1,2],[2,2]...],[[2,1],[1,1]...]... ]')
    elif len(data.shape) == 2:
        data = [data]
    else:
        raise ValueError('FEL data should be a or a serise of 2D array-like and shape should be (n,2) or (m,n,2), e.t. [[1,2],[1,3],....] or [[[1,2],[2,2]...],[[2,1],[1,1]...]... ]')
    n_figs = len(data)
    w,h = 1,1
    if n_figs > 1:
        w = 2
        h = int(ceil(n_figs/2.0))
    if cmap==None:
        cmap = cm.jet
    figure(figsize=(8*w,6*h))
    for index,d in enumerate(data):
        subplot(h,w,index+1)
        print(d.shape)
        print('\nThe x axis is:',min(d[0,:]),max(d[0,:]))
        print('\nThe y axis is:',min(d[1,:]),max(d[1,:]))
        #z,xedge, yedge = np.histogram2d(d[0,:], d[1,:], bins=bins)
        z,xedge, yedge = np.histogram2d(d[0,:], d[1,:], bins=bins , range=[[min(d[0,:])*1.05,max(d[0,:])*1.05],[min(d[1,:])*1.05,max(d[1,:])*1.05]])

        x = 0.5*(xedge[:-1] + xedge[1:])
        y = 0.5*(yedge[:-1] + yedge[1:])
        zmin_nonzero = np.min(z[np.where(z > 0)])
        z = np.maximum(z, zmin_nonzero)
        F = KbT*np.log(z)
        F -= np.max(F)
        F = np.minimum(F, 0)
        extent = [yedge[0], yedge[-1], xedge[0], xedge[-1]]
        
        print(xedge)
        print(yedge)
        contourf(x,y,F.T,17, cmap=cmap, extent=extent,levels=[i for i in range(-17,0,1)]+[0])
        clb = colorbar()
        clb.set_label('Free energy (kJ/mol)',fontsize=17)
        clb.set_ticks([i for i in range(-17,1,1)][::-1])
        xlabel("Eigenvector 1",fontsize=17)
        ylabel("Eigenvector 2",fontsize=17)
        xlim(min(d[0,:])*1.2,max(d[0,:])*1.2)
        ylim(min(d[1,:])*1.2,max(d[1,:])*1.2)
        plt.tick_params(labelsize=14)
         
        #plt.xticks([-5,-4,-3,-2,-1,0,1,2,3,4,5,6])
        tick_spacing = 2
         

    savefig('{}.eps'.format(fig),dip=1200,bbox_inches='tight')
    print('#'*40)
    print('figure saved to {}.pdf!'.format(fig))
    print('#'*40)
    return (x,y,F)

def FEL_pdf_1D(data,bins=50,cmap=None,KbT=-2.479,fig=None,xlabel_ = 'x'):
    data = np.array(data)
    if len(data.shape) == 1:
        data = [data]
    elif len(data.shape) >2 :
        raise ValueError('FEL data should be a or a serise of 1D array-like and shape should be (n) or (m,n), e.t. [1,2,3....] or [[1,2,3....],[2,3,4...]....]')
    if cmap==None:
        cmap = cm.jet
    if fig==None:
        fig = '1D_pdf_FEL'
    figure(figsize=(8,6))    
    for index,d in enumerate(data):
        z,xedge = np.histogram(d,bins=bins)
        x = 0.5*(xedge[:-1] + xedge[1:])
        zmin_nonzero = np.min(z[np.where(z > 0)])
        z = np.maximum(z, zmin_nonzero)
        F = KbT*np.log(z)
        F -= np.max(F)
        F = np.minimum(F, 0)
        plot(x,F,)
        xlim(min(x),max(x))
        ylim(min(F),max(F))
        xlabel(xlabel_,fontsize=17)
        ylabel('Free energy (kJ/mol)',fontsize=17)
    savefig('{}.png'.format(fig),dip=720,bbox_inches='tight')
    print('#'*40)
    print('figure saved to {}.png!'.format(fig))
    print('#'*40)
    return (x,F)
def find_minimal(region,x,y,z,n=1,data=None,hills='',step=500,unit=0.002):
    #region = [([x1,x1_],[y1,y1_]),([x2,x2_],[y2,y2_]),()...]
    minimal_point = []
    for r in region:
        x_index = [find_nearest_point_(x,[r[0]])[0],find_nearest_point_(x,[r[1]])[0]]
        x_index = [np.min(x_index),np.max(x_index)]
        y_index = [find_nearest_point_(y,r[2])[0],find_nearest_point_(y,r[3])[0]]
        y_index = [np.min(y_index),np.max(y_index)]
        
        posi = np.where(z == np.min(z[y_index[0]:y_index[1],x_index[0]:x_index[1]]))
        minimal_point.append([x[posi[1][0]],y[posi[0][0]]])
    points = []
    if hills !='':
        hills_data = pd.read_csv(hills,comment='#',sep='[,\t ]+',engine='python',header=None).iloc[:,[1,2]]
        for p in minimal_point:
            points.append(find_nearest_point(hills_data,p,n))
    else:
        for p in minimal_point:
            points.append(find_nearest_point(data,p,n)) #乘以步长
    print('please use gmx tools to extract structures for next time points:')
    for i,p in enumerate(points):
        print('{}: {}'.format(i,','.join([str(int(pp*step*unit)) for pp in p])))
def find_nearest_point_(seq,point):
    distance = np.abs(seq-point)
    return np.where(distance==np.min(distance))[0]
def find_nearest_point(seq,point,num=1):
    distance = np.abs(np.linalg.norm(seq-point,axis=1))
    sort_index = np.argsort(distance)
    return sort_index[:num]
def fes_FEL(data,cmap=None,xlabel_='x',ylabel_='y',fig='hills_FEL'):
    if cmap==None:
        cmap = cm.jet
    figure(figsize=(7,6))
    fes = pd.read_csv(data, sep=r'\s*', header=None, comment='#')
    # fes[0] = fes[0]/374
    # fes[1] = fes[1]/374

    xi = np.linspace(fes[0].min(), fes[0].max(), 100)
    yi = np.linspace(fes[1].min(), fes[1].max(), 100)
    zi = griddata(fes[0], fes[1], fes[2], xi, yi, interp='linear')
    #x,y,z = interpolate_EFL(xi,yi,zi,npoint=1000)
    #extent = [yedge[0], yedge[-1], xedge[0], xedge[-1]]
    # contour(xi,yi,zi, 20, linewidths = 0.1)
    contourf(xi,yi,zi, 100,cmap=cm.jet)
    clb = colorbar()
    xlabel(xlabel_,fontsize=17)
    ylabel(ylabel_,fontsize=17)
    clb.set_label('Energy (kj/mol)')
    # scatter(np.array(minimal_point).T[0],np.array(minimal_point).T[1],s=10,color='white')
    savefig('{}.png'.format(fig),dpi=720,bbox_inches='tight')
    return (xi,yi,zi)
def matplotlib_to_plotly(cmap, pl_entries):
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        # C = map(np.uint8, np.array(cmap(k*h)[:3])*255)
        C = [np.uint8(x) for x in  np.array(cmap(k*h)[:3])*255]
        pl_colorscale.append([k*h, 'rgb'+str((C[0], C[1], C[2]))])

    return pl_colorscale
def FEL_3D(data,cmap=None,xlabel_='x',ylabel_='y',fig='hills_FEL_3D'):
    if cmap==None:
        cmap = cm.jet
    figure(figsize=(7,6))
    xi,yi,zi = data
    cmap_new = matplotlib_to_plotly(cmap, 255)
    data = [
        go.Surface(x=xi,y=yi,z=zi,
                   colorscale=cmap_new,
                   colorbar=go.surface.ColorBar(title='Free energy (KJ/mol)',titleside='right',titlefont={'size':17}),
                   #lighting=dict(ambient=0.4, diffuse=0.5, roughness = 0.9, specular=0.6, fresnel=0.2),
                   lightposition = dict(x=10,y=-10,z=-200)
                   #contours=go.surface.Contours(z=go.surface.contours.Z(show=True,usecolormap=True,highlightcolor="#42f462",project=dict(z=True)))
        ),

        
    ]
    layout = go.Layout(
        title='Energy landscape',
        titlefont={'size':19},
        autosize=True,
        width=1000,
        height=1000,
        scene={
            'xaxis':{'title': 'CV1','titlefont':{'size':17},'tickfont':{'size':15}},
            'yaxis':{'title': 'CV2','titlefont':{'size':17},'tickfont':{'size':15}},
            'zaxis':{'title': '','tickfont':{'size':15}},
            },
    )
    figs = go.Figure(data=data, layout=layout)
    plot_(figs, filename='{}_3D'.format(fig),auto_open=False)

def pca_data(traj,pdb):
    try:
        top = md.load(pdb)
        featurizer = RawPositionsFeaturizer(atom_indices=top.topology.select('name CA'))
        feat = featurizer.fit_transform(traj)
    except:
        pdb = pdb_CA(pdb)
        top = md.load(pdb)
        os.remove(pdb)
        featurizer = RawPositionsFeaturizer(atom_indices=top.topology.select('name CA'))
        feat = featurizer.fit_transform(traj)
    pca = PCA()
    pca = pca.fit_transform(feat)
    data = np.array([np.vstack(pca)[:,0],np.vstack(pca)[:,1]])
    return data

def read_data(datas):
    data = []
    for d in datas:
        data.append(pd.read_csv(d,comment='#',sep='[,\t ]+',engine='python',header=None)) #sep='[,\t ]+'采用正则方式判断数据文件每列的分隔符，此处采用“,”、“tab”、“空格” 分割，其他分割方式可再添加

    new_data = []
    for d in data:
        new_data.append(d.iloc[:,-1]) #默认每个文件的最后一列作为自由能图谱的数据
    return np.array(new_data)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-xtc",default=None,nargs='+',help = '使用pca制作自由能时，需指定xtc格式的轨迹文件以及pdb文件')
    parser.add_argument("-pdb",default=None,help = '')
    group = parser.add_mutually_exclusive_group()
    group .add_argument("-pdf",help='使用概率密度函数制作自由能图谱',action='store_true')
    group .add_argument("-fes",help='使用plumed sum_hills 重构的自由能数据画出自由能图谱(2D显示)',action='store_true')
    group .add_argument("-pca",help='使用pca的前两个向量制作自由能',action='store_true')
    parser.add_argument("-FEL_3d",help='3D显示二维自由能图谱',action='store_true')
    parser.add_argument("-data",default=None,help='使用概率密度函数的制作自由能时，该参数输入第一个数据，plumd重构的自由能，用该参数指定自由能所在的文件')
    parser.add_argument("-data1",default=None,help = '使用概率密度函数制作自由能时，只指定data1时制作一维自由能，同时指定data2则制作二维自由能')
    parser.add_argument("-fig",default='FEL',help = '自由能图谱的保存路径及名称，e.t. \\test\\fig')
    parser.add_argument("-kbt",default=-2.479,help = '玻尔兹曼常数，默认2.479')
    parser.add_argument("-xy",default=None,nargs='+',help = 'xlabel/ylabel，e.t. "x y"')
    parser.add_argument("-area",default=None,type=float,nargs='+',help = '提取代表性结构的自由能有区域，按照x1 x2 y1 y2为一个区域，多个区域按同样规则放在后面，使用元动力学重构的自由能提取结构时，需同时制定HILLS文件')
    parser.add_argument("-n",default=1,type=int,help = '每个自由能区域提取几个结构，默认提取一个')
    parser.add_argument("-step",default=500,type=int,help = '用于制作自由能的数据是间隔多少步输出一次的')
    parser.add_argument("-unit",default=0.002,type=float,help = '模拟轨迹步长时间(单位ps)')
    parser.add_argument("-hills",default=None,help = 'HILLS文件')

    args = parser.parse_args()

    if args.xtc and not args.pdb:
        raise ArgumentsError('输入轨迹文件时，必须同时输入对应的pdb文件！')

    if (args.pdf or args.fes) and not args.data:
        raise ArgumentsError('采用概率密度函数或调和元动力学重构的自由能作图时，必须通过-data1指定相应的数据文件')

    if args.pca and not args.xtc:
        raise ArgumentsError('采用pca制作自由能，需输入轨迹文件以及对应的pdb文件')

    if args.pca:
        traj = load_traj(args.xtc,args.pdb)
        data = pca_data(traj,args.pdb)
        cmap = cm.colors.LinearSegmentedColormap.from_list('new_map',[cm.nipy_spectral(i) for i in range(0,256,1)]+[('white')],100)
        if args.xy:
            x,y = args.xy
        else:
            x,y = 'PC 1','PC 2'
        d = FEL_pdf_2D(data,KbT=-abs(args.kbt),fig=args.fig,xlabel_=x,ylabel_=y,cmap=cmap)

    elif args.pdf:
        data = [args.data]
        if args.data1:
            data.append(args.data1)
        data = read_data(data)
        if args.xy:
            x,y = args.xy
        else:
            x,y = 'x','y'    
        if data.shape[0] == 2:
            cmap = cm.colors.LinearSegmentedColormap.from_list('new_map',[cm.nipy_spectral(i) for i in range(0,255,1)]+[('white')],17)
            # cmap = cm.colors.LinearSegmentedColormap.from_list('new_map',[cm.nipy_spectral(i) for i in range(0,256,1)]+[('white')],15)
            d = FEL_pdf_2D(data,KbT=-abs(args.kbt),fig=args.fig,xlabel_=x,ylabel_=y,cmap=cmap)    
        else:
            d = FEL_pdf_1D(data,KbT=-abs(args.kbt),fig=args.fig,xlabel_=x)    
    elif args.fes:
        data = args.data
        if args.xy:
            x,y = args.xy
        else:
            x,y = 'CV1','CV2'    
        if args.fig==None:
            fig = 'hills_FEL'
        else:
            fig = args.fig
        d = fes_FEL(data,fig=fig,xlabel_=x,ylabel_=y)

    if args.FEL_3d:      #仅限于对二维自由能图谱制作三维显示，只输入一个数据时制作的是一维自由能图谱，再采用该步骤时会报错
        FEL_3D(d,fig=args.fig,xlabel_=x,ylabel_=y)

    area = args.area
    if area and len(area)%4==0:#提取结构需指定区域，每4个为一组指定一个区域
        area = [area[i*4:(i+1)*4]   for i in range(int(len(area)/4))]
        x,y,z = d
        if args.pdf or args.pca:
            find_minimal(area,x,y,z,data=data,n = args.n)
        elif args.fes and args.hills:
            find_minimal(area,x,y,z,hills=args.hills,n = args.n,step = args.step,unit = args.unit)
        


        















