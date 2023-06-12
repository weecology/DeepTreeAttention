library(tidyverse)
library(scales)
fils <- list.files("/Users/benweinstein/Dropbox/Weecology/Species/Metrics/", full.names = T, pattern = "metrics")
dfs<-list()
for(fil in fils){
  print(fil)
  metrics<-read.csv(fil)
  site<-str_match(fil,"(\\w+)_")[,2]
  train<-read.csv(paste("/Users/benweinstein/Dropbox/Weecology/Species/Metrics/",site,"_train.csv",sep=""))
  species_counts<-train %>% group_by(individual) %>% slice(1) %>% 
    group_by(taxonID) %>% summarize(n=n())
  species_counts$accuracy<-NA
  species_counts$precision<-NA
  for (x in species_counts$taxonID){
      accuracy<-metrics[metrics$Metric==paste("accuracy_",x,sep=""),"Value"]
      if(length(accuracy)==0){
        next
      }
      species_counts[species_counts$taxonID==x,"accuracy"] <- accuracy
      species_counts[species_counts$taxonID==x,"precision"] <-metrics[metrics$Metric==paste("precision_",x,sep=""),"Value"]
  }
  species_counts$precision<-as.numeric(species_counts$precision)
  species_counts$accuracy<-as.numeric(species_counts$accuracy)
  species_counts$site<-site
  
  #Rank order
  species_counts<-species_counts %>% arrange(desc(n))
  species_counts$rank<-1:nrow(species_counts)
  dfs[[site]]<-species_counts
}

#Break by habitat type

habitat<-data.frame(
  site=c("BONA","DEJU","GRSM","NIWO","SERC","SJER","STEI","TREE","WREF","UNDE","DELA","LENO",
         "OSBS","JERC","TALL","CLBJ","TEAK",
         "SOAP","YELL","MLBS","BLAN","BART","HARV","UKFS"),
  Habitat=c("Conifer","Conifer","Southern Broadleaf","Conifer",
            "Southern Broadleaf","Savannah","Northern Broadleaf","Northern Broadleaf",
            "Conifer","Northern Broadleaf","Southern Broadleaf","Southern Broadleaf","Savannah",
            "Savannah","Southern Broadleaf","Savannah","Conifer","Conifer","Conifer","Southern Broadleaf",
            "Southern Broadleaf","Northern Broadleaf","Northern Broadleaf","Southern Broadleaf"))
df<-bind_rows(dfs)
df<-na.omit(df)

#how many species?
length(unique(df$taxonID))

# Data but not in the models
a<-read.csv("/Users/benweinstein/Dropbox/Weecology/Species/Metrics/annotations.csv")
s<-unique(a$taxonID)
a %>% filter(taxonID %in% s[!s %in% df$taxonID]) %>% group_by(taxonID) %>% summarize(n=n()) %>% arrange(desc(n)) %>% as.data.frame()

#example black cherry
a %>% filter(taxonID=="PRSE2") %>% group_by(siteID) %>% summarize(n=n())
a %>% filter(taxonID=="PRSES") %>% group_by(siteID,filename) %>% summarize(n=n())

# red spruce
a %>% filter(taxonID=="PIRU") %>% group_by(siteID,filename) %>% summarize(n=n())

#Confusing ones
a %>% filter(taxonID=="JUVIV") %>% group_by(siteID) %>% summarize(n=n())
a %>% filter(taxonID=="JUVI") %>% group_by(siteID) %>% summarize(n=n())

df<-df %>% inner_join(habitat)
ggplot(df,aes(x=rank, y=accuracy, color=site)) + 
  geom_point(size=1) + labs(x="Rank Order of Training Samples") + 
  facet_wrap(~Habitat,scales="free") + geom_smooth(aes(group=1),method="glm",method.args=list(family="binomial"))

ggsave("/Users/benweinstein/Dropbox/Weecology/Species/Metrics/rank_order.png",height=4,width=7)

ggplot(df,aes(x=log(n), y=accuracy, color=site)) + 
  geom_point(size=2) + geom_line(aes(group=site)) +
  scale_x_continuous("Training Samples") + facet_wrap(~Habitat)

#Overall accuracy
fils <- list.files("/Users/benweinstein/Dropbox/Weecology/Species/Metrics/", full.names = T, pattern = "metrics")
fil_list<-lapply(fils, function(x){
  g<-read.csv(x)
  site<-str_match(x,"(\\w+)_")[,2]
  g<-pivot_wider(g, names_from = "Metric", values_from = "Value")
  g<-g[,c("overall_micro","overall_macro")]
  g$site<-site
  train<-read.csv(paste("/Users/benweinstein/Dropbox/Weecology/Species/Metrics/",site,"_train.csv",sep=""))
  species_counts<-train %>% group_by(individual) %>% slice(1) 
  g$train_samples<-nrow(species_counts)
  g$num_species<-train %>% group_by(taxonID) %>% summarize(n=n()) %>% nrow() 
  g$overall_macro<-as.numeric(g$overall_macro)
  g$overall_micro<-as.numeric(g$overall_micro)
  return(g)
})
table_frame<-bind_rows(fil_list)

table_frame<-table_frame %>% select(site, num_species, train_samples, overall_micro,overall_macro)
write_csv(table_frame, "/Users/benweinstein/Dropbox/Weecology/Species/Metrics/output_table.csv")


# Rank prediction abundance
csvs<-list.files("/Users/benweinstein/Dropbox/Weecology/Species/SpeciesMaps/Abundances", full.names=T)
abundances<-lapply(csvs, read.csv)
names(abundances) <- sapply(csvs, function(x){ str_match(x, "/(\\w+)_species")[,2]})
for (x in 1:length(abundances)){
  abundances[[x]]$site<-names(abundances)[x]
  abundances[[x]]$proportion <-  abundances[[x]]$scientific/sum(abundances[[x]]$scientific)
  abundances[[x]] <- abundances[[x]] %>% arrange(desc(proportion)) %>% mutate(rank=1:nrow(.))
}

abundances<-bind_rows(abundances)
abundances<-abundances %>% inner_join(habitat)
ggplot(abundances, aes(x=rank, y=proportion, col=site)) + 
  geom_point(size=1) + geom_line() +
  facet_wrap(~Habitat, scales="free_x") +
  labs(y="Predictions", x="Rank",col="Site") +
  scale_y_continuous(labels = scales::percent) + 
  guides(col=guide_legend(ncol=2))

ggsave("/Users/benweinstein/Dropbox/Weecology/Species/Metrics/rank_abundance.png",height=5,width=7)

# Number of individuals
sum(abundances$scientific)

# Number of species
length(unique(abundances$X))


       