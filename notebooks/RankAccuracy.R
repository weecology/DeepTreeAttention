library(tidyverse)
library(scales)
library(sf)
library(meltr)
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
      accuracy<-metrics[metrics$name==paste("accuracy_",x,sep=""),"valueCurrent"]
      if(length(accuracy)==0){
        next
      }
      species_counts[species_counts$taxonID==x,"accuracy"] <- accuracy
      species_counts[species_counts$taxonID==x,"precision"] <-metrics[metrics$name==paste("precision_",x,sep=""),"valueCurrent"]
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
         "SOAP","YELL","MLBS","BLAN","BART","HARV","UKFS","RMNP"),
  Habitat=c("Conifer","Conifer","Southern Broadleaf","Conifer",
            "Southern Broadleaf","Savannah","Northern Broadleaf","Northern Broadleaf",
            "Conifer","Northern Broadleaf","Southern Broadleaf","Southern Broadleaf","Savannah",
            "Savannah","Southern Broadleaf","Savannah","Conifer","Conifer","Conifer","Southern Broadleaf",
            "Southern Broadleaf","Northern Broadleaf","Northern Broadleaf","Southern Broadleaf","Conifer"))
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
  geom_point(aes(size=n)) + labs(x="Rank Order of Training Samples") + 
  facet_wrap(~Habitat,scales="free") + 
  geom_smooth(aes(group=1),method="glm",method.args=list(family="binomial")) 
ggsave("/Users/benweinstein/Dropbox/Weecology/Species/Metrics/rank_order.png",height=6,width=10)

ggplot(df,aes(x=rank, y=accuracy, color=site,size=n)) + 
  geom_point() + geom_line(alpha=0.1) + 
  geom_label(aes(label=site)) +
  labs(x="Rank Order of Training Samples") +
  facet_wrap(~Habitat,scales="free")

ggplot(df,aes(x=log(n), y=accuracy, color=site)) + 
  geom_point(size=2) + geom_line(aes(group=site)) +
  scale_x_continuous("Training Samples") + facet_wrap(~Habitat)

#Overall accuracy
fils <- list.files("/Users/benweinstein/Dropbox/Weecology/Species/Metrics/", full.names = T, pattern = "metrics")
fil_list<-lapply(fils, function(x){
  g<-read.csv(x)
  site<-str_match(x,"(\\w+)_")[,2]
  g<- g %>% filter(name %in% c("overall_micro","overall_macro")) %>% select(name, valueCurrent) %>% spread(name, valueCurrent)
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

mean(table_frame$overall_micro)
mean(table_frame$overall_macro)

table_frame<-table_frame %>% select(site, num_species, train_samples, overall_micro,overall_macro)
write_csv(table_frame, "/Users/benweinstein/Dropbox/Weecology/Species/Metrics/output_table.csv")

# Rank prediction abundance
csvs<-list.files("/Users/benweinstein/Dropbox/Weecology/Species/SpeciesMaps/Abundances", full.names=T)
abundances<-lapply(csvs, read.csv)
names(abundances) <- sapply(csvs, function(x){ str_match(x, "/(\\w+)_abundance")[,2]})
for (x in 1:length(abundances)){
  abundances[[x]]$site<-names(abundances)[x]
  abundances[[x]]$proportion <- abundances[[x]]$count/sum(abundances[[x]]$count)
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
sum(abundances$count)

# Number of species
length(unique(abundances$sci_name))
length(unique(abundances$site))

# Create site x species matrix
all_points<-read_sf("/Users/benweinstein/Dropbox/Weecology/Species/SpeciesMaps/crops/canopy_points.shp")
taxonomy<-read.csv("/Users/benweinstein/Documents/DeepTreeAttention/data/raw/OS_TAXON_PLANT-20220330T142149.csv")
taxonomy<-taxonomy %>% select(scientificName, taxonID,taxonRank) %>% mutate(sci_name=stringr::word(scientificName,0,2)) %>% filter(taxonRank=="species")
sitexspp<-data.frame(table(abundances$sci_name,abundances$site))
colnames(sitexspp)<-c("sci_name","site","present")
rawspeciesmatrix<-all_points %>% group_by(taxonID,siteID) %>% filter(n() > 1) %>% group_by(taxonID, siteID) %>%
  slice(1) %>% select(taxonID, siteID) %>% merge(taxonomy, "taxonID") %>% mutate(sci_name =scientificName)

sites <- unique(abundances$site)
results_list <- list()
for (x in sites){
  species_present <- sitexspp %>% filter(site == x,present==1)
  species_in_raw <- rawspeciesmatrix %>% filter(siteID==x) %>% distinct(sci_name, taxonID)
  raw_stems<-all_points %>% filter(siteID==x, taxonID %in% species_in_raw$taxonID)
  included <- species_in_raw[species_in_raw$sci_name %in% species_present$sci_name,] 
  missing <- species_in_raw[!species_in_raw$sci_name %in% species_present$sci_name,] 
  proportion_present <- nrow(species_present)/nrow(species_in_raw)
  stems_of_present_species<-raw_stems %>% filter(taxonID %in% included$taxonID)
  stems_present<-nrow(stems_of_present_species)/nrow(raw_stems)
  results_list[[x]] <- data.frame(site=x, Species=proportion_present, Stems=stems_present)
}

results<-bind_rows(results_list)
pdf<-gather(results,"var",count,2:3)
ggplot(pdf,aes(y=site,x=count,col=var)) + geom_point(size=4) + theme_bw() + scale_color_manual(values=c("grey","black")) + 
  scale_x_continuous() + scale_x_continuous(labels = scales::percent) +
  labs(x="Proportion of known species included in model",color="Aggregated by") +
  geom_vline(linetype="dashed",xintercept=mean(results$Species), color="grey",size=1.5) +
  geom_vline(linetype="dashed",xintercept=mean(results$Stems), color="black",size=1.5) 
  
ggsave("/Users/benweinstein/Dropbox/Weecology/Species/Metrics/proportion_included.png",height=6,width=10)

site_lists<-all_points %>% group_by(taxonID,siteID) %>%
  filter(n() > 1) %>% group_by(taxonID, siteID) %>% 
  select(taxonID, siteID) %>% merge(taxonomy, "taxonID") %>% 
  group_by(sci_name, siteID) %>% summarize(n=n()) %>% as.data.frame() %>% select(-geometry) %>% group_by(siteID) %>% arrange(siteID,desc(n))

#label which are in the models.
site_lists<-sitexspp %>% mutate(Included_in_Model=present,siteID=site) %>% select(-present, -site) %>% merge(site_lists, by=c("sci_name","siteID"))  %>% arrange(siteID,desc(n))

write.csv(site_lists,"/Users/benweinstein/Dropbox/Weecology/Species/SpeciesMaps/site_lists.csv")



