# Classification of Alzheimer MRI Images By Keras CNN Model
 Classification of T1-Weighted MRI images of Alzheimer's Disease, in Python. Obtained from an open-source dataset. Supervised classification is done by designing a CNN model by using the Keras library. :x_ray:
<br>
<br>

August 2025 - Emirhan & Oguzhan

<br><br>

:tr: Türkçe: Derin Öğrenme ile Alzheimer MR görüntülerinin sınıflandırılması. Aksiyal eksende T1-W MR görüntüleri çekilerek hazırlanmış olan, erişime açık bir veri seti kullanılmıştır. Keras kütüphanesi ile tasarlanmış bir CNN modeli kullanılarak, bu görüntüler başarıyla sınıflandırılmıştır. :x_ray:

<br><br>

## Content
1) Dataset Description
2) Code Organization
3) Current Results



<br><br>



## 1) Dataset Description

 OASIS-1 DATASET (2007) : Alzheimer's Disease (AD) Brain MRI Images <br>

 Inspired and utilized by: 
 [Github_Repo](https://github.com/O-Memis/AlzheimerMRI) <br>

 Original source: [OasisBrains](https://sites.wustl.edu/oasisbrains/home/oasis-1/) <br><br><br>



**Data characteristics:** 

 T1-W protocol of MRI images in Transverse plane. <br>
    
 Cross-sectional data collection of 461 subjects, 18-96 yrs. <br>
    
 86k images (2D slices) converted to .JPG from NIFTI by Kaggle data providers. <br>
    
 8-bit depth and [248, 496] ratio of **gray-scale** images, but saved in 3 channels. <br><br>

 The classes are UNBALANCED! The Non-Demented class dominates the others. <br><br>
    
 There are **4 classes** to classify AD: <br>
                                        **non-demented** <br>
                                        **mild-demented** <br>
                                        **modereate demented** <br>
                                        **very demented** <br>

<br><br><br>


## 2) Current Results

 Test **Accuracy**: %90  buraya gerçek skoru yazalım <br>


 <br><br>
 5-Fold Cross-Validation Results: <br>
 Average **Accuracy**: %90  çapraz doğrulama skoru <br>




<br><br><br>
:mailbox: Contact me at  blabla@gmail.com <br> I'm waiting for your suggestions and contributions.
