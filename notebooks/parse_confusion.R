#Parse confusion matrices, download confusion matrix from coment under 'assets' for a given experiment
library(jsonlite)
library(reshape2)
library(bipartite)
library(dplyr)
library(tidyr)

parse_file<-function(path){
  a<-fromJSON(path)
  label_names <- a[3]$labels
  #matrix of predictions
  m <- a[4]$matrix
  colnames(m)<-label_names
  rownames(m)<-label_names
  return(m)
}
metadata<-parse_file("/Users/Ben/Downloads/MetadataMatrix")
HSI<-parse_file("/Users/Ben/Downloads/HSIMatrix")
RGB<-parse_file("/Users/Ben/Downloads/RGBMatrix")
ensemble<-parse_file("/Users/Ben/Downloads/Ensemble Matrix")
#nested temp 0 = perfectly nested
nestedness(HSI, RGB)$temperature


#which are not nested
mHSI<-melt(HSI)
colnames(mHSI)<-c("true","predicted","count")
mHSI$Method<-"HSI"

mRGB<-melt(RGB)
colnames(mRGB)<-c("true","predicted","count")
mRGB$Method<-"RGB"

sum(ensemble)/
mensemble<-melt(ensemble)
colnames(mensemble)<-c("true","predicted","count")
mensemble$Method<-"ensemble"

df<-bind_rows(list(mRGB,mHSI,mensemble))

#Accuracy
sum_total = sum(HSI)
df %>% group_by(Method) %>% filter(!count==0,predicted==true) %>% summarize(n=sum(count)/sum_total)

#HSI improve
species_count <- df %>% filter(Method=="HSI") %>% group_by(true) %>% summarize(total=sum(count))
df %>% group_by(Method,true) %>% filter(!count==0,predicted==true) %>% summarize(n=sum(count)) %>%
  pivot_wider(names_from = Method, values_from = n) %>% mutate(HSI_improve=HSI-RGB) %>%  inner_join(species_count) %>%
    mutate(p=100*HSI_improve/total) %>%
    arrange(desc(p))

#RGB improve
species_count <- df %>% filter(Method=="HSI") %>% group_by(true) %>% summarize(total=sum(count))
df %>% group_by(Method,true) %>% filter(!count==0,predicted==true) %>% summarize(n=sum(count)) %>%
  pivot_wider(names_from = Method, values_from = n) %>% mutate(HSI_improve=HSI-RGB) %>%  inner_join(species_count) %>%
  mutate(p=100*HSI_improve/total) %>%
  arrange(p)

