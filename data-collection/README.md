# Data Collection
### Data Source 
All the data for this project is taken from Kaggle competition [avito-demand-prediction](https://www.kaggle.com/c/avito-demand-prediction/).The data consists of images and csv files.
### Data Transfer 
Data from Kaggle was tranferred to Google DataStore Bucket by the following steps: 
1) Virtual Machiene with over 100 Gi of Memory was spun up in Google Cloud Platform.
2) ssh to the VM from your local machiene.
3) Install kaggle api and authenticate
4) Download one zipped file from Kaggle and unzip it.
5) Transfer the unzipped csv/image file/folder to Google datastore.
6) Delete the files on the virtual machiene when the transfer is complete

Repeat steps 4,5 and 6 for all the files.

The complete data tranfer took around 16 hours.

### Data Size

After unzipping all the files, total size of the dataset is about 100 GigaBytes.

| File | Size  |
| :---   | :- |
| **periods_train.csv** | 776 MB |
| **test.csv** | 331 MB |
| **test_active.csv** | 8.5 GB |
| **train.csv** | 1 GB |
| **train_active.csv** |  9.2 GB |
| **Images Test** | 20 GB |
| **Images Train** | 50 GB |


### Dataset
The dataset consists of the below files:

**train.csv** - Train data with the following columns. 

| Column | Description  |
| :---   | :- |
| item_id | Unique Ad id |
| user_id | User id |
| region | Region where the item is sold |
| parent_category_name | Top level ad category as classified by Avito's ad model |
| category_name |  Fine grain ad category as classified by Avito's ad model |
| param_1 | Optional parameter from Avito's ad model |
| param_2 | Optional parameter from Avito's ad model |
| param_3 | Optional parameter from Avito's ad model |
| title | Ad title |
| description | Ad description |
| price | Selling price|
| item_seq_number | Ad sequential number for user |
| activation_date | Date ad was activated |
| user_type | User type category |
| image | Id code of image. Ties to a jpg file in train_jpg. Not every ad has an image |
| image_top_1 | Avito's classification code for the image |
| deal_probability | The target variable. This is the likelihood that an ad actually sold something |


**test.csv** - Test data. Same schema as the train data, minus deal_probability.

**train_active.csv** - Supplemental data from ads that were displayed during the same period as train.csv. Same schema as the train data minus deal_probability, image, and image_top_1.

**test_active.csv** - Supplemental data from ads that were displayed during the same period as test.csv. Same schema as the train data minus deal_probability, image, and image_top_1.

**periods_train.csv** - Supplemental data showing the dates when the ads from train_active.csv were activated and when they where displayed.

| Column | Description  |
| :---   | :- |
| item_id | Ad id. Maps to an id in train_active.csv. IDs may show up multiple times in this file if the ad was renewed |
| activation_date | Date the ad was placed |
| date_from | First day the ad was displayed |
| date_to | Last day the ad was displayed |


**periods_test.csv** - Supplemental data showing the dates when the ads from test_active.csv were activated and when they where displayed. Same schema as periods_train.csv, except that the item ids map to an ad in test_active.csv.

**train_jpg.zip** - Images from the ads in train.csv.

**test_jpg.zip** - Images from the ads in test.csv.

**train_jpg_{0, 1, 2, 3, 4}.zip** - These are the exact same images as you'll find in train_jpg.zip but split into smaller zip archives so the data are easier to download. If you already have train_jpg.zip you do NOT need to download these. We have not made these zips available in kernels as they would only increase the kernel creation time.




