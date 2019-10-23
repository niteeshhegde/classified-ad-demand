# Data Collection
### DataSource
All the data for this project is available in Kaggle. 

Its consists of images and csv files.

The data is first downloaded using kaggle-api into the local system. 

It was unzipped and then transfered to Google DataStore bucket for further operations.


**train.csv** - Train data. 

item_id - Ad id. 

user_id - User id.

region - Ad region.

city - Ad city.

parent_category_name - Top level ad category as classified by Avito's ad model.

category_name - Fine grain ad category as classified by Avito's ad model.

param_1 - Optional parameter from Avito's ad model.

param_2 - Optional parameter from Avito's ad model.

param_3 - Optional parameter from Avito's ad model.

title - Ad title.

description - Ad description.

price - Ad price.

item_seq_number - Ad sequential number for user.

activation_date- Date ad was placed.

user_type - User type.

image - Id code of image. Ties to a jpg file in train_jpg. Not every ad has an image.

image_top_1 - Avito's classification code for the image.

deal_probability - The target variable. This is the likelihood that an ad actually sold something. It's not possible to verify every transaction with certainty, so this column's value can be any float from zero to one.

**test.csv** - Test data. Same schema as the train data, minus deal_probability.

**train_active.csv** - Supplemental data from ads that were displayed during the same period as train.csv. Same schema as the train data minus deal_probability, image, and image_top_1.

**test_active.csv** - Supplemental data from ads that were displayed during the same period as test.csv. Same schema as the train data minus deal_probability, image, and image_top_1.

**periods_train.csv** - Supplemental data showing the dates when the ads from train_active.csv were activated and when they where displayed.

item_id - Ad id. Maps to an id in train_active.csv. IDs may show up multiple times in this file if the ad was renewed.

activation_date - Date the ad was placed.

date_from - First day the ad was displayed.

date_to - Last day the ad was displayed.

**periods_test.csv** - Supplemental data showing the dates when the ads from test_active.csv were activated and when they where displayed. Same schema as periods_train.csv, except that the item ids map to an ad in test_active.csv.

**train_jpg.zip** - Images from the ads in train.csv.

**test_jpg.zip** - Images from the ads in test.csv.

**train_jpg_{0, 1, 2, 3, 4}.zip** - These are the exact same images as you'll find in train_jpg.zip but split into smaller zip archives so the data are easier to download. If you already have train_jpg.zip you do NOT need to download these. We have not made these zips available in kernels as they would only increase the kernel creation time.

***Datasource link*** - https://kaggle.com/c/avito-demand-prediction/data

### DataSize
After unzipping all the files, total size of the dataset is about 100 GigaBytes.

**periods_test.csv** 632 MB

**periods_train.csv** 776 MB

**test.csv** 331 MB

**test_active.csv** 8.5 GB

**train.csv** 1 GB

**train_active.csv** 9.2 GB

**Images Test** 20 GB

**Images Train** 50 GB
