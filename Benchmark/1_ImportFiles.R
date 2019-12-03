# Import datasets from OpenML

library("OpenML")

dataset_ids = c(3, 24, 31, 38, 44, 179, 715, 718, 720, 722, 723, 727, 728, 734, 735, 737, 740, 741, 819, 821, 822, 823, 833, 837, 843, 845, 846, 847)
setOMLConfig(apikey = "c1994bdb7ecb3c6f3c8f3b35f4b47f1f")
dir.create('data')

df = data.frame()
for (ds_id in dataset_ids)
{
  dataset = getOMLDataSet(data.id = ds_id)
  print(dim(dataset$data))
  where_target = which(dataset$colnames.old==dataset$target.features)
  if (where_target!=ncol(dataset$data)) print("ERROR")
  name_df = dataset$desc$name
  num_rows = dim(dataset$data)[1]
  num_cols = dim(dataset$data)[2]
  df = rbind(df, data.frame(name_df,num_rows,num_cols,where_target))
  write.csv(dataset$data,file=paste0('data/',name_df,'.csv'),row.names = FALSE)
}
write.csv(df,file='bases_datos.csv',row.names = FALSE)

