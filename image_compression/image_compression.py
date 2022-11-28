import cv2
import numpy as np
import imageio
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm

def compress_image(image, k, save_name='compressed_image.gif'):
	"""
	Compresses an image using SVD and saves it as a GIF.
	:param image: path of image to compress
	:param k: number of singular values/modes to keep
	:param save_name: name/path of the output GIF
	:return: None
	"""
	#read the image
	img = cv2.imread(image)

	#convert BGR to RGB
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	#split the image into 3 channels
	red, green, blue = cv2.split(img)
	
	#perform singular value decomposition on each channel
	u_red, s_red, v_red = np.linalg.svd(red)
	u_green, s_green, v_green = np.linalg.svd(green)
	u_blue, s_blue, v_blue = np.linalg.svd(blue)
	
	images = []
	save_list = [] #save the images at these k values

	#plot the singular values and the cumulative sum of the singular values
	fig, ax = plt.subplots(1, 2, figsize=(15, 5))
	ax[0].plot(s_red, label='Red')
	ax[0].plot(s_green, label='Green')
	ax[0].plot(s_blue, label='Blue')
	ax[0].set_xlabel('Singular Values')
	ax[0].set_ylabel('Value')
	ax[0].set_title('Singular Values')
	ax[0].legend()
	ax[1].plot(np.cumsum(s_red), label='Red')
	ax[1].plot(np.cumsum(s_green), label='Green')
	ax[1].plot(np.cumsum(s_blue), label='Blue')
	ax[1].set_xlabel('Singular Values')
	ax[1].set_ylabel('Value')
	ax[1].set_title('Cumulative Sum of Singular Values')
	ax[1].legend()
	plt.show()

	print("Program will proceed after graphs have been closed...")

	for i in tqdm(range(k), desc='Generating Compressed Images'):
		#reconstruct the image using up to the kth singular value for each channel 
		red_reconstructed = np.dot(u_red[:, :i], np.dot(np.diag(s_red[:i]), v_red[:i, :]))
		green_reconstructed = np.dot(u_green[:, :i], np.dot(np.diag(s_green[:i]), v_green[:i, :]))
		blue_reconstructed = np.dot(u_blue[:, :i], np.dot(np.diag(s_blue[:i]), v_blue[:i, :]))
		
		#adding text to a white image to indicate the number of singular values used to reconstruct the image
		white = np.ones((red_reconstructed.shape[0], red_reconstructed.shape[1], 3), np.uint8) * 255
		cv2.putText(white, '# Of Singular Values Used = ' + str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
		
		#combine the channels to form the final image
		img_reconstructed = cv2.merge((red_reconstructed, green_reconstructed, blue_reconstructed))

		#adding the text on the white image to the bottom of the reconstructed image
		img_reconstructed = np.concatenate((img_reconstructed, white), axis=0)

		#crop off the excess white space
		img_reconstructed = img_reconstructed[:-(red_reconstructed.shape[0]-50), :, :]

		#save the reconstructed image
		images.append(img_reconstructed)

		#save the compressed image from the save_list. you will need
		#to create a folder called 'out' if you are saving the images
		if i in save_list:
			imageio.imwrite(str(path)+"/out/"+save_name+str(i)+".png", img_reconstructed)

	images = [np.uint8(img) for img in images] #supressing warnings by converging float64 to uint8
	print("Please wait while the GIF is being generated...")
	imageio.mimsave(save_name, images)

path = pathlib.Path().absolute()
compress_image(str(path) + '/simple.jpg', 200, save_name='simple.gif')
