
# Read results
dataset_ids = c(3, 24, 31, 38, 44, 179, 715, 718, 720, 722, 723, 727, 728, 734, 735, 737, 740, 741, 819, 821, 822, 823, 833, 837, 843, 845, 846, 847)
tabla_resultados_ids = c(179,24,3,31,38,44,715,718,720,722,723,727,728,734,735,737,740,741,819,821,822,823,833,837,843,845,846,847)
df = read.csv("bases_datos.csv")
df = cbind(df,dataset_ids)
df2 = data.frame()
errores = c()
NFs = c()
minutes_total = c()
for (id_tabla in tabla_resultados_ids)
{
  numrow = which(dataset_ids==id_tabla)
  db_name = df[numrow,'name_df']
  df2 = rbind(df2, df[numrow,])
  print(db_name)
  tst_error = c()
  nfs_v = c()
  minutes = c()
  for (i in 1:10)
  {
    file_name = paste0('resultados/sec/GAparsimony_',db_name,'_iter_',i,'.RData')
    if (file.exists(file_name))
    {
      load(file_name)
      tst_error = c(tst_error, -GAparsimony_model@bestfitnessTst)
      nfs_v = c(nfs_v,  round(GAparsimony_model@bestcomplexity/1e6))
      minutes = c(minutes, GAparsimony_model@minutes_total)
    }
  }
  if (length(tst_error)>0) 
  {
    print(paste0("ERROR MEDIO=",mean(tst_error, na.rm=TRUE)))
    errores = c(errores, mean(tst_error, na.rm=TRUE))
    NFs = c(NFs, mean(nfs_v, na.rm=TRUE))
    minutes_total = c(minutes_total, mean(minutes, na.rm=TRUE))
  } else {
    errores = c(errores, 0.0)
    NFs = c(NFs, 0.0)
    minutes_total = c(minutes_total, 0.0)
  }
}

df2 = cbind(df2, errores, NFs, minutes_total)
df3 = read.csv("tabla_mljar.csv")
df_final = cbind(df3,df2)
df_final$num_cols = df_final$num_cols-1
df_final = df_final[,c(1,6,7,2,3,4,11,8,12,13)]
colnames(df_final) = c("id","name",'#rows','Autoklearn','H2O','MLJAR',"GAparsimony",'#inputs',"NFS","minutes")
df_final
write.csv(df_final,"df_final.csv")
