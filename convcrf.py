#------------------------------------------------------------------------------
#  Libraries
#------------------------------------------------------------------------------
import torch
import numpy as np
from convcrf import convcrf


#------------------------------------------------------------------------------
#  ConvCRF
#------------------------------------------------------------------------------
class ConvCRF(object):
	def __init__(self, num_classes, height, width, kernel_size=7, pyinn=False, normalize=False):
		self.num_classes = num_classes
		self.normalize = normalize

		self.config = convcrf.default_conf
		self.config['filter_size'] = kernel_size
		self.config['pyinn'] = pyinn

		self.gausscrf = convcrf.GaussCRF(conf=self.config, shape=(height, width), nclasses=num_classes)
		self.gausscrf.cuda()

	def __call__(self, image, mask):
		with torch.no_grad():

			# One-hot mask
			mask = (np.arange(self.num_classes) == mask[...,None])

			if self.normalize:
				# Warning, applying image normalization affects CRF computation.
				# The parameter 'col_feats::schan' needs to be adapted.

				# Normalize image range
				#     This changes the image features and influences CRF output
				image = image / 255
				# mean substraction
				#    CRF is invariant to mean subtraction, output is NOT affected
				image = image - 0.5
				# std normalization
				#       Affect CRF computation
				image = image / 0.3

				# schan = 0.1 is a good starting value for normalized images.
				# The relation is f_i = image * schan
				self.config['col_feats']['schan'] = 0.1

			# Make input pytorch compatible
			image = image.transpose(2, 0, 1)
			image = np.expand_dims(image, axis=0)
			img_var = torch.from_numpy(image.astype('float32')).cuda()

			unary = mask.transpose(2, 0, 1)
			unary = np.expand_dims(unary, axis=0)
			unary_var = torch.from_numpy(unary.astype('float32')).cuda()

			# Perform CRF inference
			prediction = self.gausscrf.forward(unary=unary_var, img=img_var)
			prediction = prediction[0].data.cpu().numpy()
			result = np.argmax(prediction, axis=0)
			return result


#------------------------------------------------------------------------------
#  Main execution
#------------------------------------------------------------------------------
if __name__ == "__main__":
	import cv2
	from time import time

	image_file = "cc6e5e804.jpg"
	label_file = "mask.png"

	image = cv2.imread(image_file)[...,::-1]
	label = cv2.imread(label_file, 0)

	crf = ConvCRF(num_classes=5, height=256, width=1600)

	start_time = time()
	result = crf(image, label)
	print("Execution time: %.2f [ms]" % (1000*(time()-start_time)))

	result = 255 * (result==1).astype('uint8')
	cv2.imwrite("out.png", result)
