import numpy as np 
from  glob import glob
import SimpleITK as sitk
import os
import nibabel as nib 
import matplotlib.pyplot as plt 
 
import pandas as pd

'''
sitk读取过来的dicom数组,是CT值,pydicom读取的是像素值
numpy.transpose会复制一份
numpy=浅拷贝


todo: 再生成几个典型的,
如5mm-1mm
1.25mm-1mm
10mm-1mm

'''

def read_cts(source_dicom_dir):
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()# 加载公开元信息
    reader.LoadPrivateTagsOn()# 加载私有元信息
    img_names = reader.GetGDCMSeriesFileNames(source_dicom_dir)
    #print(img_names)
    reader.SetFileNames(img_names)
    image = reader.Execute()# 读取ct序列
    # sitk读取的为ct值矩阵
    cts_origin  = sitk.GetArrayFromImage(image) # z, y, x
    return cts_origin
    
def data_argument(image):  # [ n, h, w]
    '''
    数据增强函数,因为后面换了tensorflow的环境,所以这个用torch写的增强函数暂时搁置了
    
    '''
    return image


def get_interpolator_nmae(index):
    '''index:int,取值范围为1-15的整数,本方法用于返回重采样时使用的插值方法对应的方法名'''
    if(index>15|index<0|isinstance(index,int)):
        print('Error:index不在规定范围内(1-15)或者类型不是int')
        return None
    methods_name={}
    methods_name[1]='sitkNearestNeighbor'
    methods_name[2]='sitkLinear'
    methods_name[3]='sitkBSpline'
    methods_name[4]='sitkGaussian'
    methods_name[5]='sitkLabelGaussian'
    methods_name[6]='sitkHammingWindowedSinc'
    methods_name[7]='sitkCosineWindowedSinc'
    methods_name[8]='sitkWelchWindowedSinc'
    methods_name[9]='sitkLanczosWindowedSinc'
    methods_name[10]='sitkBlackmanWindowedSinc'
    # methods_name[11]='sitkBSplineResampler'
    methods_name[11]='sitkBSplineResamplerOrder3'
    methods_name[12]='sitkBSplineResamplerOrder1'
    methods_name[13]='sitkBSplineResamplerOrder2'
    methods_name[14]='sitkBSplineResamplerOrder4'
    methods_name[15]='sitkBSplineResamplerOrder5'
    return methods_name[index]

def resample_image(sitk_imgs, out_spacing=[1.0, 1.0, 2.0],method=sitk.sitkBSpline,):
    '''
    sitk_imgs:SimpleITK中的Image类型的图像,也就是待重采样的原始图像.
    out_spacing:长度为3的数值列表.指定重采样之后的图像在X,Y,Z轴的像素值大小.
    method:整数,取值范围1-15. 使用get_interpolator_nmae获取对应的插值名字
    '''
    original_spacing=sitk_imgs.GetSpacing()	# [(0.884766, 0.884766, 5.0)]
    original_size = sitk_imgs.GetSize() # [H,W,N] 
    
    # 根据输出out_spacing设置新的size
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(sitk_imgs.GetDirection())
    resample.SetOutputOrigin(sitk_imgs.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(sitk_imgs.GetPixelIDValue())
 
    resample.SetInterpolator(method)# sitk.sitkBSpline
    return resample.Execute(sitk_imgs) 

def set_window(img,window_center=-400,window_width=1500,slope=1,intercept=-1024):
    '''
    img:ndarray, 原始图像数组.建议是[N,H,W]大小,如[34,512,512].
    window_center:float, 调窗过后窗位.
    window_width:float, 调窗过后窗宽.
    slope:float, 像素矩阵映射到CT值的时候斜率.
    intercept:float, 像素矩阵映射到CT值的时候截距.
    特别提醒:如果是用sitk打开dicom文件夹,那么得到的矩阵是CT矩阵而不是像素矩阵,也就是说不用再进行像素矩阵到CT值矩阵的转换了

    '''
    # data=img.copy()
    data=img
    # data=slope*data+intercept# sitk 读取出来的,会自动调整成CT值..pydicom读取的就不会
    minWindow=window_center-window_width/2.0

    data=(data-minWindow)/window_width 
    data[data<0]=0
    data[data>1]=1
    # 调窗操作代码可以优化
    data=(data*255).astype('uint8')#(data*255).astype('uint8')
    return  data 

def resample(source_dicom_dir,target_dicom_dir,file_name,thickness,
        file_type='npy',method=sitk.sitkBSpline,
        adjust_window=True,window_center=-300,window_width=1200,slope=1,intercept=-1024,
        slice=False,slice_number=48):
    '''这个函数是用来进行3维dicom影像重建的函数.

    source_dicom_dir:String,需要进行重建的原始dicom文件所在文件夹,如./PA0/ST0/SE0.

    target_dicom_dir:String,重建后的nii.gz文件保存的文件夹,该文件夹不存在时会自动创建.

    file_name:String,重建后文件名,不带文件格式后缀.如file_type为npy时,保存成file_name.npy文件

    thickness:float,重建后的图像厚度(重建后的图像在x,y,z轴上的像素实际值(单位:mm).[x_spacing,y_spacing,z_spacing].

    file_type:String, 重建后数据保存的格式,可选:['npy','nii'],默认npy格式.

    method:list or int , 重建过程中使用的插值方法.
 
    adjust_window: boolean, 重建后的图像是否进行调窗操作(请注意,本方法先进行?重建,再进行调窗?,如果设置调窗的话).

    window_center: int,调整后窗位(肺部一般是-300).

    window_width:  int,调整后窗宽(肺部一般是1200).

    slope:int, 根据像素值pixel计算CT值(HU),公式slpoe*pixel+intercept.

    intercept:int, 根据像素值pixel计算CT值(HU),公式slpoe*pixel+intercept.

    slice:boolean, 是否需要对重建后的图像进行切片操作.

    slice_number:int, 重建后图像切片后数目.

    slice_method:String, 切片方法.
    {1: 'sitkNearestNeighbor', 2: 'sitkLinear', 3: 'sitkBSpline', 4: 'sitkGaussian', 5: 'sitkLabelGaussian', 6: 'sitkHammingWindowedSinc', 7: 'sitkCosineWindowedSinc', 8: 'sitkWelchWindowedSinc', 9: 'sitkLanczosWindowedSinc', 10: 'sitkBlackmanWindowedSinc', 11: 'sitkBSplineResamplerOrder3', 12: 'sitkBSplineResamplerOrder1', 13: 'sitkBSplineResamplerOrder2', 14: 'sitkBSplineResamplerOrder4', 15: 'sitkBSplineResamplerOrder5'}
    '''
    reader = sitk.ImageSeriesReader()

    if not os.listdir(source_dicom_dir):
        print(f"路径不存在! source_dicom_dir:{source_dicom_dir}")
        return None

    series_names=reader.GetGDCMSeriesFileNames(source_dicom_dir)
    if len(series_names)==0:
        print('输入文件夹内无Dicom 文件序列,请检查!当前输入路径为:',source_dicom_dir)
        return None

    file_types= ['npy','nii','none']
    if file_type not in file_types:
        print('当前指定输出文件格式不符合规范,请重新选定! 可选:', "".join(file_types))
        return None

    if not os.path.exists(target_dicom_dir):
        os.makedirs(target_dicom_dir)

    reader.SetFileNames(series_names )

    # 设置读取到的CT值矩阵格式,//todo 如果这里不加呢,忘记验证了
    reader.SetOutputPixelType(sitk.sitkInt16)
     
    sitk_imgs = reader.Execute()
    # print('sitk的到数组的shape:',sitk.GetArrayFromImage(sitk_imgs).shape)
    # print('sitk得到的数组的类型',sitk.GetArrayFromImage(sitk_imgs).dtype)
    #print(sitk.GetArrayFromImage(sitk_imgs).sum())
    # 保存原始的dicom文件系列
    # sitk.WriteImage(
    #     sitk_imgs,os.path.join(target_dicom_dir,'0.nii.gz')
    #     )	
    # 保存原始的调窗后的文件
    # nib.save(nib.Nifti1Image(set_window(sitk.GetArrayFromImage(sitk_imgs).transpose(2,1,0)[::-1,::-1,::-1],window_center,window_width,slope,intercept), np.eye(4)) , os.path.join(target_dicom_dir,'0_w.nii.gz') )
    
    sp=list(sitk_imgs.GetSpacing())# 得到在x,y,z轴上的spacing 
    if  sp[-1]!= thickness:#如果当前ct厚度和给定的不一致,那么进行重采样操作
         
        sp[-1]=thickness # 设置厚度
        # 先重采样
        sitk_imgs=resample_image(sitk_imgs,sp,method) 
    # 重采样前后数组的类型不变
    resampled_imgs_arr=sitk.GetArrayFromImage(sitk_imgs)

    # print(f'重采样之后数组的类型:{resampled_imgs_arr.dtype}')
    # print(f'重采样之后数组的shape::{resampled_imgs_arr.shape}')

    if adjust_window:# 重采样之后需要调窗,在调窗输出的时候也要设定类型
        adjwin_imgs_arr=set_window(resampled_imgs_arr,window_center,window_width,slope,intercept)
    else:
        adjwin_imgs_arr=resampled_imgs_arr
    # print(f'保存成数组的shape:{adjwin_imgs_arr.shape}')#CHW
    # print(f'保存成数组的dtype:{adjwin_imgs_arr.dtype}')

    if slice:# 如果需要进行标准化操作,即产生一系列切片
        # 1. 当前切片方式采用中心化切片,即只取中心的那部分切片
        start=int((adjwin_imgs_arr.shape[0]-slice_number)/2)
        adjwin_imgs_arr=adjwin_imgs_arr[start:start+slice_number]

   
    if file_type=='nii':    
        nib.save(nib.Nifti1Image( adjwin_imgs_arr.transpose(2,1,0)[::-1,::-1,::-1] , np.eye(4)) , os.path.join(target_dicom_dir,file_name+'.nii.gz'))
    elif file_type=='npy':
        adjwin_imgs_arr.save(filename=os.path.join(target_dicom_dir,file_name+'.npy'))
    elif file_type=='sitk':
        return sitk_imgs
    elif file_type=='none':
        return adjwin_imgs_arr

  
 


if __name__=="__main__":
    
    a=resample(source_dicom_dir='./TimesCTS3/cuijiongshu/ST0/thin/SE00',target_dicom_dir='./test',file_name='none',thickness=1,
        file_type='none',method=sitk.sitkBSpline,
        adjust_window=True,window_center=-300,window_width=1200,slope=1,intercept=-1024,
        slice=False,slice_number=48)
    for i,ct in enumerate(a):
        plt.figure()
        plt.imshow(ct,cmap='gray')
        plt.savefig('./test2/'+str(i)+'_.png')
        plt.close()
    for i,ct in enumerate(a):
        plt.figure()
        plt.imshow(ct,cmap='gray')
        plt.savefig('./test/'+str(i)+'.png')
        plt.close()
    a=(np.array(a)+1024)
    
    


    
    


