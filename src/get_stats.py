import pickle

with open('seg_feature_stats.txt', 'rb') as file:
    results = pickle.loads(file.read()) # use `pickle.loads` to do the reverse

for key, value in results.items():
	print(value['dice_mean'])
	print(value['l1_diff'][0])
# print(results)