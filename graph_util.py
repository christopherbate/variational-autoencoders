
# # Setup live plotting.
# fig = plt.figure()
# ax = []
# im_plots = []
# for i in range(test_vector.shape[0]):
#     random_data = tf.random.normal(shape=(28, 28))
#     subax = fig.add_subplot(4, 4, i+1)
#     ax.append(subax)
#     img_plot = subax.imshow(random_data)
#     im_plots.append(img_plot)
# plt.ion()
# plt.show()