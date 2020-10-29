# Distorted-Image-Classification
Image classification using CNN on distorted grey scale images







This project is created to classify 28000 grey scale images.
The images are distorted handwritten digits in grey scale. The input supplied to this is in the form of excel.
The excel has the first column as label and the next 784(28*28) columns as pixels ranging from 0 to 255 in value.

This input format has been reshaped to 28 28 1 to failitate model actions. Normalisation of these pixels were much needed to generalise the value range on these pixels.
A CNN network has been used here with split validation.

The model reaches validation accuracy of approximate 98 percent.
