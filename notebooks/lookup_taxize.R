library(dplyr)
library(NeonTreeEvaluation)
library(taxize)
library(stringr)
a<-read.csv("/Users/benweinstein/Downloads/train (35).csv")
table(a$taxonID)
length(unique( a$taxonID))
matching_names<-field %>% filter(taxonID %in% a$taxonID) %>% distinct(scientificName, taxonID)
cleaned_matching_names <- as.character(matching_names$scientificName)
cleaned_matching_names<-sapply(cleaned_matching_names, function(x) { word(x, 1, 2)})
db<-classification(cleaned_matching_names,"tropicos", ask=FALSE,rows=1, return_id=TRUE)
families<-sapply(db, function(x){
  if(length(x) ==1){
    return(NA)
  } else{
    return(x[x$rank=="subclass","name"])
  }
  })

result<-data.frame(scientificName=matching_names$scientificName,families)
result[result$scientificName=="Magnolia sp.","families"]="Magnoliidae"

result<-result %>% inner_join(matching_names)

missing_ids = unique(a$taxonID)[!unique(a$taxonID) %in% result$taxonID]

#Make sure we aren't missing any lookups
missing<-c("ABAM","ASTR","THOC2","QUPA2","LALA","ACCI","PIRE","ABLA","PICL","ABCO","ABAM","THPL","HATE3","MAAC","ILCA","TADI","MAGNO")
missing_families = c("Pinidae","Magnoliidae","Pinidae",
                     "Magnoliidae",'Pinidae',"Magnoliidae","Pinidae","Pinidae","Pinidae",
                     "Pinidae","Pinidae","Pinidae","Magnoliidae","Magnoliidae","Magnoliidae","Pinidae","Magnoliidae")
missing_df <-data.frame(taxonID=missing, families=missing_families)
a %>% filter(!taxonID %in% field$taxonID) %>% distinct(taxonID)
result <- bind_rows(list(result, missing_df))
write.csv(result, "/Users/benweinstein/Documents/DeepTreeAttention/data/raw/families.csv")

