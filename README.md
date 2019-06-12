# Handwritten-Digit-Recognition-using-SVM
                                                   1. Introduction
Hand writing recognition of characters has been around since the 1980s.The task of handwritten digit recognition, using a classifier, has great importance and use such as – online handwriting recognition on computer tablets, recognize zip codes on mail for postal mail sorting, processing bank check amounts, numeric entries in forms filled up by hand (for example ‐ tax forms) and so on. There are different challenges faced while attempting to solve this problem. The handwritten digits are not always of the same size, thickness, or orientation and position relative to the margins. Our goal was to implement a pattern classification method to recognize the handwritten digits provided in the MNIST dataset of images of hand written digits (0 ‐ 9). The data set used for our application is composed of 300 training images and 300 testing images, and is a subset of the MNIST data set (originally composed of 60,000 training images and 10,000 testing images). Each image is a 28 x 28 grayscale (0 ‐ 255) labeled representation of an individual digit. The general problem we predicted we would face in this digit classification problem was the similarity between the digits like 1 and 7, 5 and 6, 3 and 8, 9 and 8 etc. Finally the uniqueness and variety in the handwriting of different individuals also influences the formation and appearance of the digits.

Problems with handwritten digits:

The handwritten digits are not always of the same size, width, orientation and justified to margins as they differ from writing of person to person, so the general problem would be while classifying the digits due to the similarity between digits such as 1 and 7, 5 and 6, 3 and 8, 2 and 5, 2 and 7, etc. This problem is faced more when many people write a single digit with a variety of different handwritings. Lastly, the uniqueness and variety in the handwriting of different individuals also influence the formation and appearance of the digits. Now we introduce the concepts and algorithms of deep learning and machine learning.

The algorithm used here is Support vector machine (SVM) which  is a learning method based on statistical learning theories. Basing on the principle of structural risk minimization, SVM can improve the generalization ability of the learning machine as much as possible. Even the decision rules obtained from limited training samples can still get small errors for independent test datasets. In recent years, SVM has been widely used in pattern recognition, regression analysis and feature extraction. 



1.2  Hardware Requirements

Processor – Intel i5.
Hard Disk – 500 GB.
Memory – 4 GB.

1.3 Software Requirements
   1.3.1 PyCharm
PyCharm is the most popular IDE for Python, and includes great features such as excellent code completion and inspection with advanced debugger and support for web programming and various frameworks. As any other IDE editor, it supports basic features like bookmarks, breakpoints, syntax highlighting, code completion, zooming, folding code blocks, etc. There are, however, plenty of advanced features like macros, highlighted TODO items, code analysis, intention actions, intelligent and fast navigation, and a lot more.
PyCharm is created by Czech company, Jet brains which focusses on creating integrated development environment for various web development languages like JavaScript and PHP.

 1.3.2 Qt Designer

Qt Designer is the Qt tool for designing and building graphical user interfaces (GUIs) with Qt Widgets. You can compose and customize your windows or dialogs in a what-you-see-is-what-you-get (WYSIWYG) manner, and test them using different styles and resolutions.

Widgets and forms created with Qt Designer integrate seamlessly with programmed code, using Qt's signals and slots mechanism, so that you can easily assign behavior to graphical elements. All properties set in Qt Designer can be changed dynamically within the code. Furthermore, features like widget promotion and custom plugins allow you to use your own components with Qt Designer.

 We have the option of using Qt Quick for user interface design rather than widgets. It is      a much easier way to write many kinds of applications. It enables a completely customizable appearance, touch-reactive elements, and smooth animated transitions, backed up by the power of OpenGL graphics acceleration






                                        2. SYSTEM REQUIREMENT SPECIFICATION
                                        
                                        
2.1. PURPOSE
The purpose of this Software Requirement Documentation is to provide high-level and detailed descriptions of “Handwritten Digit Recognition”. This Software Requirement Documentation will provide quantifiable requirements of the android for use by the designer and the users of this project.
.

TECHNOLOGY USED.

2.2.1 PyQt5:
Qt is a set of C++ libraries and development tools that includes platform independent abstractions for graphical user interfaces, networking, threads, regular expressions, SQL databases, SVG, OpenGL, XML, user and application settings, positioning and location services, short range communications (NFC and Bluetooth), web browsing, 3D animation, charts, 3D data visualisation and interfacing with app stores. PyQt5 implements over 1000 of these classes as a set of Python modules.
PyQt5 supports the Windows, Linux, UNIX, Android, macOS and iOS platforms. PyQt does not include a copy of Qt. You must obtain a correctly licensed copy of Qt  yourself. However, binary wheels of the GPL version of PyQt5 are provided and these include a copy of the appropriate parts of the LGPL version of Qt.


   
2.2.2 Skicit-learn:
Scikit-learn (formerly scikits.learn) is a free software machine learning library for the Python programming language.[3] It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means and DBSCAN, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy.
The scikit-learn project started as scikits.learn, a Google Summer of Code project by David Cournapeau. Its name stems from the notion that it is a "SciKit" (SciPy Toolkit), a separately-developed and distributed third-party extension to SciPy. The original codebase was later rewritten by other developers. In 2010 Fabian Pedregosa, Gael Varoquaux, Alexandre Gramfort and Vincent Michel, all from INRIA took leadership of the project and made the first public release on February the 1st 2010.Of the various scikits, scikit-learn as well as scikit-image were described as "well-maintained and popular" in November 2012.
Scikit-learn is largely written in Python, with some core algorithms written in Cython to achieve performance. Support vector machines are implemented by a Cython wrapper around LIBSVM; logistic regression and linear support vector machines by a similar wrapper around LIBLINEAR. 

2.2.3 NumPy:
NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. The ancestor of NumPy, Numeric, was originally created by Jim Hugunin with contributions from several other developers. In 2005, Travis Oliphant created NumPy by incorporating features of the competing Numarray into Numeric, with extensive modifications. NumPy is open-source software and has many contributors. 

2.2.4     OpenCV:

OpenCV (Open Source Computer Vision Library) is an open source computer vision and machine learning software library. OpenCV was built to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in the commercial products. Being a BSD-licensed product, OpenCV makes it easy for businesses to utilize and modify the code. The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms. These algorithms can be used to detect and recognize faces, identify objects, classify human actions in videos, track camera movements, track moving objects, extract 3D models of objects, produce 3D point clouds from stereo cameras, stitch images together to produce a high resolution image of an entire scene, find similar images from an image database, remove red eyes from images taken using flash, follow eye movements, recognize scenery and establish markers to overlay it with augmented reality, etc. OpenCV has more than 47 thousand people of user community and estimated number of downloads exceeding 14 million. The library is used extensively in companies, research groups and by governmental bodies. Along with well-established companies like Google, Yahoo, Microsoft, Intel, IBM, Sony, Honda, Toyota that employ the library, there are many startups such as Applied Minds, VideoSurf, and Zeitera, that make extensive use of OpenCV.  OpenCV’s deployed uses span the range from stitching streetview images together, detecting intrusions in surveillance video in Israel, monitoring mine equipment in China, helping robots navigate and pick up objects at Willow Garage, detection of swimming pool drowning accidents in Europe, running interactive art in Spain and New York, checking runways for debris in Turkey, inspecting labels on products in factories around the world on to rapid face detection in Japan.
It has C++, Python, Java and MATLAB interfaces and supports Windows, Linux, Android and Mac OS. OpenCV leans mostly towards real-time vision applications and takes advantage of MMX and SSE instructions when available. A full-featured CUDA and OenCL interfaces are being actively developed right now. There are over 500 algorithms and about 10 times as many functions that compose or support those algorithms. OpenCV is written natively in C++ and has a templated interface that works seamlessly with STL containers.

USER INTERFACE REQUIREMENT

PyQT5 Image Viewer: selected image will be displayed here, in order to get predicted.

Text Box: the predicted digit will be displayed in a textbox widget of android application.

Select Button: button which on being selected opens the file dialogue box to select images.
    Predict Button: button which on being selected predicts the digit for a given image according   to the training model.


                                          3.  PROPOSED ALGORITHM

  3.1. Algorithm
The algorithm being used for training the model is support vector machine classifier.A SVM is a discriminative classifier formally defined by a seperating hyperplane.In two dimensional  space this hyperplane is  a line dividing a plane in two parts where in each class lay other side.

The objective of the support vector machine algorithm is to find a hyperplane in an N-dimensional space(N — the number of features) that distinctly classifies the data points.

To separate the two classes of data points, there are many possible hyperplanes that could be chosen. Our objective is to find a plane that has the maximum margin, i.e the maximum distance between data points of both classes. Maximizing the margin distance provides some reinforcement so that future data points can be classified with more confidence.


Support vectors are data points that are closer to the hyperplane and influence the position and orientation of the hyperplane. Using these support vectors, we maximize the margin of the classifier. Deleting the support vectors will change the position of the hyperplane. These are the points that help us build our SVM.

3.2 Dataset Used
The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. It was created by "re-mixing" the samples from NIST's original datasets. The creators felt that since NIST's training dataset was taken from American Census Bureau employees, while the testing dataset was taken from American high school students, it was not well-suited for machine learning experiments. Furthermore, the black and white images from NIST were normalized to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels.

The MNIST database contains 60,000 training images and 10,000 testing images. Half of the training set and half of the test set were taken from NIST's training dataset, while the other half of the training set and the other half of the test set were taken from NIST's testing dataset.

Division of the MNIST took place by 30,000 samples from SD-3 and 30,000 samples from SD- 1 with 250 writers approx. and 5,000 samples from SD-3 and remaining 5,000 samples from SD-1 to form a different test set. Images of digits were taken from various scanned digits, normalized in size and justify as centered. This makes it an excellent dataset for evaluating models and allowing the machine learning aspirant to focus on deep learning and machine learning with very little data cleaning.



    
                                  4.  SYSTEM DESIGN AND PROJECT IMPLEMENTATION



4.3 Training the Classifier

 Training of classifier is done in following steps:
Calculate the HOG features for each sample in the database.
Train a multi-class linear SVM with the HOG features of each sample along with the corresponding label.
Save the classifier in a file .



4.3.1 Calculating HOG features
We create an numpy array containing the HOG features which will be used to train the classifier. To calculate the HOG features, we set the number of cells in each block equal to one and each individual cell is of size 14×14. Since our image is of size 28×28, we will have four blocks/cells of size 14×14 each. Also, we set the size of orientation vector equal to 9. So our HOG feature vector for each sample will be of size 4×9 = 36.
4.3.2 Histogram of Oriented Gradients (HOG)
The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. The technique counts occurrences of gradient orientation in localized portions of an image. This method is similar to that of edge orientation histograms, scale-invariant feature transform descriptors, and shape contexts, but differs in that it is computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy.
The essential thought behind the histogram of oriented gradients descriptor is that local object appearance and shape within an image can be described by the distribution of intensity gradients or edge directions. The image is divided into small connected regions called cells, and for the pixels within each cell, a histogram of gradient directions is compiled. The descriptor is the concatenation of these histograms. For improved accuracy, the local histograms can be contrast-normalized by calculating a measure of the intensity across a larger region of the image, called a block, and then using this value to normalize all cells within the block. This normalization results in better invariance to changes in illumination and shadowing. 
The HOG descriptor has a few key advantages over other descriptors. Since it operates on local cells, it is invariant to geometric and photometric transformations, except for object orientation. Such changes would only appear in larger spatial regions. Moreover, as Dalal and Triggs discovered, coarse spatial sampling, fine orientation sampling, and strong local photometric normalization permits the individual body movement of pedestrians to be ignored so long as they maintain a roughly upright position. The HOG descriptor is thus particularly suited for human detection in images.

4.4 Testing the Classifier

We perform the following steps for testing the classifier:
Load the test image and convert it to a grayscale image using cv2.cvtColor function.
We then apply a Gaussian filter to the grayscale image to remove noisy pixels.
Convert the grayscale image into a binary image.
Calculating the contours in the image and then calculating the bounding box for each contour.
Calculating the contours in the image and then calculating the bounding box for each contour.
 Generate a bounding square around each contour.
 Then resize each bounding square to a size of 28×28 and dilate it.
 Calculate the HOG features for each bounding square.
Predict the image using trained classifier, HOG feature vector for each bounding square should be of the same size for which the classifier was trained.


                                             6. CONCLUSION AND FUTURE SCOPE
                                          
We presented a system for dealing with such problem. The system started by acquiring an image containing digits, this image was digitized using some optical devices and after applying some enhancements and modifications to the digits within the image it can be recognized using feed forward back propagation algorithm. The studies were conducted on the Arabic handwriting digits of 10 independent writers who contributed a total of 1300 isolated Arabic digits these digits divided into two data sets: Training 1000 digits, testing 300 digits. An overall accuracy meet using this system was 65% on the test data set used.

We developed a system for Arabic handwritten recognition. And we efficiently choose a segmentation method to fit our demands. Our system successfully designs and implement a neural network which efficiently go without demands, after that the system are able to understand the Arabic numbers that was written manually by users.

Recently handwritten digit recognition becomes vital scope and it is appealing many researchers because of its using invariety of machine learning and computer vision applications.


                                                    5. REFERENCES
                                                     
http://yann.lecun.com/exdb/mnist/

Saeed AL- Mansoori  Intelligent Handwritten Digit Recognition using Artificial Neural

Network Int. Journal of Engineering Research and Applications Vol. 5, Issue 5, May 2015

Bellili, M. Gilloux, P. Gallinari  An MLP-SVM combination architecture for offline handwritten digit recognition

https://www.kaggle.com/learn/machine-learning

Handwritten digit recognition by multi class Support vector machine: by Yu Wang


Study on  Handwritten digit recognition by support vector machine: IOP Conference Series: Materials Science and Engineering by Xiaoning Zhou.

Vladimir N. Vapnik An Overview of Statistical Learning Theory. IEEE TRANSACTIONS ON NEURAL NETWORKS, VOL. 10, NO. 5, SEPTEMBER 1999.

Vladimir N.Vapnik.The Nature of Statistical Learning Theory, Springer, N Y, (1995).

Stitson M O et al .Theory of support vector machines, Technical Report CSD-T R-96-17, Depar tment of Computer Science, Roy al Holloway University of London ,(1996).

Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

C. Kaynak. Methods of Combining Multiple Classifiers and Their Applications to Handwritten Digit Recognition, MSc Thesis, Institute of Graduate Studies in Science and Engineering, Bogazici University (1995).

Guyon I, Boser B E, Vapnik V. Automatic Capacity Tuning of Very Large VC-dimension Classifiers[J]. Advances in Neural Information Processing Systems, 1992, 5:147--155.

Yu Ying, Wang Xiaolong, etc. Efficient SVM-based  Recognition of Chinese Personal Names. High Technology Letters. 2004

Hyun-Chul Kim, Shaoning Pang, Hong-Mo le, Daijin Kim,Sung Yang Bang. Constructing support vector machine ensemble. Pattem Recognition 36 (2003) 2757 – 2767 



