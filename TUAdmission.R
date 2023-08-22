#Admission Project
#by Tim Mose

#Call relevant packages from library
library(dplyr)
library(tidyverse)
library(caret)
library(e1071)
library(MASS)
library(glmnet)
library(randomForest)
library(gbm)
library(DMwR2) 
library(tree)

getwd()
#Load original data into R.
clean_data <- read.csv("TU.csv")
#After skimming through the data (view(clean_data)), 
#you notice that there are many blanks.
#First, make blanks read as NA. 
#Then, decide what to do with NA for each individual variable.
#How to handle NAs for a variables depends on the nature of that variable.
clean_data[clean_data == ''] <- NA


length(clean_data)
#There are 69 total columns. Only 46 have been cleaned already

#######Data Cleaning########
#Column1 - ID
sum(is.na(clean_data$ID))#No NA.
#ID should be removed in the modeling stage.

#Column2 - train.test
sum(is.na(clean_data$train.test))#No NA.
levels(factor(clean_data$train.test))#No suspicious categories.
#train.test should be removed in the modeling stage.

#Column3 - Entry.Term..Application.
sum(is.na(clean_data$Entry.Term..Application.))#No NA.
levels(factor(clean_data$Entry.Term..Application.))#No suspicious categories.
clean_data$Entry.Term..Application. <- as.factor(clean_data$Entry.Term..Application.)
#turning into numeric to make it easier to work with. 
clean_data$Entry.Term..Application. <- as.numeric(substring(clean_data$Entry.Term..Application., 6))

#Column4 - Admit.Type
sum(is.na(clean_data$Admit.Type))#No NA.
levels(factor(clean_data$Admit.Type))#has only one level!!
#Since the data set only has first years (i.e.,only one category), 
#Admit.Type should be removed. - it is possible that all other types (transfers were not included)

#Column5 - Permanent.Postal
sum(is.na(clean_data$Permanent.Postal))#162 NAs.
#At first, I thought about doing some research to categorize 
#postal codes into different states. For instance, US-Texas
#US-Georgia, US-California, International, etc.
#However, I suddenly realized that the column "Permanent.Geomarket" 
#had already done what I want to do!
#So, we may just use "Permanent.Geomarket" and Column5 may be removed.


#Column6 - Permanent.Country
sum(is.na(clean_data$Permanent.Country))#1 NA.
levels(factor(clean_data$Permanent.Country))#No suspicious categories.
#The ID with NA in Permanent.Country is 11148. 
#Since this person is a US citizen in the column "Citizenship.Status",
#Unites States is assigned to NA.
clean_data$Permanent.Country[is.na(clean_data$Permanent.Country)] <- "United States"
clean_data$Permanent.Country <- as.factor(clean_data$Permanent.Country)
#There are way to many categories that show up just a few times. These end up making it hard to work with. Furthermore, you cannot dummycol these
#since it would lead to ~100 extra columns, which would make processing take a lot longer.
#Instead I will reclassify by continent
clean_data$United_States <- ifelse(clean_data$Permanent.Country == "United States", 1, 0)
clean_data$NorthAmerica_not_US <- ifelse(clean_data$Permanent.Country %in% c("Canada", "Mexico", "Barbados", "Cayman Islands", "Dominica", "Dominican Republic", "Jamaica", "The Bahamas", "Trinidad and Tobago"), 1, 0)
clean_data$Europe <- ifelse(clean_data$Permanent.Country %in% c("Albania", "Belgium", "Bosnia and Herzegovina", "Cyprus", "Czech Republic", "France", "Georgia", "Germany", "Greece", "Iceland", "Ireland", "Italy", "Lithuania", "Luxembourg", "Montenegro", "Netherlands", "Norway", "Poland", "Portugal", "Romania", "Russia", "Spain", "Switzerland", "Turkey", "Ukraine", "United Kingdom"), 1, 0)
clean_data$South_America <- ifelse(clean_data$Permanent.Country %in% c("Argentina", "Belize", "Bolivia", "Brazil", "Chile", "Colombia", "Costa Rica", "Ecuador", "El Salvador", "Guatemala", "Honduras", "Nicaragua", "Panama", "Paraguay", "Peru", "Uruguay", "Venezuela"), 1, 0)
clean_data$Asia <- ifelse(clean_data$Permanent.Country %in% c("Bangladesh", "Cambodia", "China", "Hong Kong S.A.R.", "India", "Indonesia", "Iran", "Japan", "Jordan", "Kazakhstan", "Kenya", "Kuwait", "Lebanon", "Malaysia", "Mongolia", "Nepal", "Oman", "Pakistan", "Palestine", "Philippines", "Saudi Arabia", "Singapore", "South Korea", "Taiwan", "Thailand", "United Arab Emirates", "Uzbekistan", "Vietnam","Australia", "New Zealand"), 1, 0)
clean_data$Oceania <- ifelse(clean_data$Permanent.Country %in% c("Australia", "New Zealand"), 1, 0)
clean_data$Africa <- ifelse(clean_data$Permanent.Country %in% c("Cote D'Ivoire", "Cameroon", "Egypt", "Ethiopia", "Ghana", "Morocco", "Mozambique", "Nigeria", "South Africa", "Tanzania", "Uganda", "Zimbabwe"), 1, 0)
#I included Oceanic countries under asia because otherwise it caused issues with various models


#Column7 - Sex
sum(is.na(clean_data$Sex))#No NA.
levels(factor(clean_data$Sex))#No typos.
clean_data$Sex <- as.factor(clean_data$Sex)

#Column8 - Ethnicity
sum(is.na(clean_data$Ethnicity))#227 NAs.
levels(factor(clean_data$Ethnicity))#No questionable category.
#I personally think it is more reasonable to impute NAs  with "Not specified"
#because when I fill out a form (e.g. in a clinic), I was usually given the
#the option of not specifying my ethnicity and/or race. 
clean_data$Ethnicity[is.na(clean_data$Ethnicity)] <- "Not specified"
clean_data$Ethnicity <- as.factor(clean_data$Ethnicity)

#Column9 - Race
sum(is.na(clean_data$Race))#555 NAs.
#Impute NAs with "Not specified". The reason is similar to that for Ethnicity.
clean_data$Race[is.na(clean_data$Race)] <- "Not specified"
levels(factor(clean_data$Race))#No questionable category.
table(clean_data$Race)
#Notice that the current classification of Race is too detailed,
#leading to very low frequencies for some categories.
#Need to consider combining some of the categories because
#a category with a small number of cases won't have 
#a significant effect on the response.
clean_data$Race <- ifelse(clean_data$Race == "American Indian or Alaska Native", "American Indian or Alaska Native",
                          ifelse(clean_data$Race == "American Indian or Alaska Native, White", "American Indian or Alaska Native, White",
                                 ifelse(clean_data$Race == "Asian", "Asian",
                                        ifelse(clean_data$Race == "Asian, White", "Asian, White",
                                               ifelse(clean_data$Race == "Black or African American", "Black or African American",
                                                      ifelse(clean_data$Race == "Black or African American, White", "Black or African American, White",
                                                             ifelse(clean_data$Race == "Not specified", "Not specified",
                                                                    ifelse(clean_data$Race == "White", "White", "Other"))))))))
clean_data$Race <- as.factor(clean_data$Race)

#Column10 - Religion
sum(is.na(clean_data$Religion))#5483 NAs.
levels(factor(clean_data$Religion))#No questionable category.
#Because no religion and other are already included in current levels,
#it is more reasonable to impute NAs with "Not specified".
clean_data$Religion[is.na(clean_data$Religion)] <- "Not specified"
table(clean_data$Religion)
#Religion has lots of options, with some options having a very small number
#of cases. 
#To the best of my knowledge in religion (please correct me if I am wrong), 
#I combine similar levels into one level.
#Then I combine levels with less than 100 cases into "Other" because
#a level accounting for low than 1% of training set is very unlikely to have
#a significant effect on the response.
clean_data$Religion <- ifelse(clean_data$Religion == "Anglican", "Anglican",
                              ifelse(clean_data$Religion == "Baptist", "Baptist",
                                     ifelse(clean_data$Religion == "Bible Churches", "Christian",
                                            ifelse(clean_data$Religion == "Buddhism", "Buddhism",
                                                   ifelse(clean_data$Religion == "Christian", "Christian",
                                                          ifelse(clean_data$Religion == "Christian Reformed", "Christian",
                                                                 ifelse(clean_data$Religion == "Christian Scientist", "Christian",
                                                                        ifelse(clean_data$Religion == "Church of Christ", "Christian",
                                                                               ifelse(clean_data$Religion == "Church of God", "Christian",
                                                                                      ifelse(clean_data$Religion == "Hindu", "Hindu",
                                                                                             ifelse(clean_data$Religion == "Islam/Muslim", "Islam/Muslim",
                                                                                                    ifelse(clean_data$Religion == "Jewish", "Jewish",
                                                                                                           ifelse(clean_data$Religion == "Lutheran", "Lutheran",
                                                                                                                  ifelse(clean_data$Religion == "Methodist", "Methodist",
                                                                                                                         ifelse(clean_data$Religion == "Not specified", "Not specified",
                                                                                                                                ifelse(clean_data$Religion == "Non-Denominational", "Non-Denominational",
                                                                                                                                       ifelse(clean_data$Religion == "None", "None", 
                                                                                                                                              ifelse(clean_data$Religion == "Presbyterian", "Presbyterian",
                                                                                                                                                     ifelse(clean_data$Religion == "Presbyterian Church of America", "Presbyterian",
                                                                                                                                                            ifelse(clean_data$Religion == "Roman Catholic", "Roman Catholic", "Other"))))))))))))))))))))
clean_data$Religion <- as.factor(clean_data$Religion)

#Column11 - First_Source.Origin.First.Source.Date
sum(is.na(clean_data$First_Source.Origin.First.Source.Date))#No NA.
clean_data$First_Source.Origin.First.Source.Date <- as.Date(clean_data$First_Source.Origin.First.Source.Date, 
                                                            format="%m/%d/%Y")

date_list <- as.POSIXlt(clean_data$First_Source.Origin.First.Source.Date, format = "%Y-%m-%d")
day_of_year_first <- date_list$yday + 1

clean_data$day_of_year_first <- day_of_year_first 
clean_data$day_of_year_first[is.na(clean_data$day_of_year_first)] <- mean(clean_data$day_of_year_first, na.rm = TRUE)

#I will also create a year column
clean_data$year_first <- as.numeric(format(clean_data$First_Source.Origin.First.Source.Date, "%Y"))
clean_data$year_first[is.na(clean_data$year_first)]<- mean(clean_data$year_first, na.rm = TRUE)


#Column12 - Inquiry.Date
sum(is.na(clean_data$Inquiry.Date))#4579 NAs.
#I will deal with NAs later because I need to use this variable to create
#several new variables.
clean_data$Inquiry.Date <- as.Date(clean_data$Inquiry.Date, format="%m/%d/%Y")
clean_data$Inquiry.Date

date_list <- as.POSIXlt(clean_data$Inquiry.Date, format = "%Y-%m-%d")
day_of_year <- date_list$yday + 1
print(day_of_year)
clean_data$day_of_year <- day_of_year 
clean_data$day_of_year[is.na(clean_data$day_of_year)] <- mean(clean_data$day_of_year, na.rm = TRUE)

#I will also create a year column
clean_data$year <- as.numeric(format(clean_data$Inquiry.Date, "%Y"))
clean_data$year[is.na(clean_data$year)]<- mean(clean_data$year, na.rm = TRUE)
sum(is.na(clean_data$year))
clean_data$year

#Column13 - 
sum(is.na(clean_data$Submitted))#No NA.
clean_data$Submitted <- as.Date(clean_data$Submitted, format="%m/%d/%Y")
date_list <- as.POSIXlt(clean_data$Submitted, format = "%Y-%m-%d")
day_of_year_first <- date_list$yday + 1

clean_data$day_of_year_first <- day_of_year_first 
clean_data$day_of_year_first[is.na(clean_data$day_of_year_first)] <- mean(clean_data$day_of_year_first, na.rm = TRUE)

#I will also create a year column
clean_data$year_first <- as.numeric(format(clean_data$Submitted, "%Y"))
clean_data$year_first[is.na(clean_data$year_first)]<- mean(clean_data$year_first, na.rm = TRUE)

#Column11-13
#After viewing Column11-13, it would be interesting to see
#whether the differences between submission date and First_Source date
#the differences between submission date and inquiry date affect the response.
#The time difference between submission date and first_source date.
clean_data$Submit_FirstSource <- difftime(clean_data$Submitted, 
                                          clean_data$First_Source.Origin.First.Source.Date, 
                                          units = "weeks")
clean_data$Submit_Inquiry <- difftime(clean_data$Submitted, 
                                      clean_data$Inquiry.Date, units = "weeks")
clean_data$Submit_FirstSource <- round(clean_data$Submit_FirstSource, digits = 0)
clean_data$Submit_FirstSource <- as.numeric(clean_data$Submit_FirstSource)
clean_data$Submit_Inquiry <- round(clean_data$Submit_Inquiry, digits = 0)
clean_data$Submit_Inquiry <- as.numeric(clean_data$Submit_Inquiry)
#Remember that there are NAs in Inquiry.Date, 
#thus leading to NAs in Submit_Inquiry.
#Impute NAs in Submit_Inquiry with median values.
clean_data$Submit_Inquiry[is.na(clean_data$Submit_Inquiry)] <- median(clean_data$Submit_Inquiry,
                                                                      na.rm=TRUE)
#Remember to remove Column11-13 in the modeling stage
#since they are used to construct new variables.   

#Column14 - Application.Source
sum(is.na(clean_data$Application.Source))#No NA.
table(clean_data$Application.Source)#No questionable category.
clean_data$Application.Source <- as.factor(clean_data$Application.Source)

#Column15 - Decision.Plan
sum(is.na(clean_data$Decision.Plan))#No NA.
table(clean_data$Decision.Plan)#No questionable category.
clean_data$Decision.Plan <- as.factor(clean_data$Decision.Plan)    

#Column16 - Staff.Assigned.Name
#Based on variable description,
#I don't think this variable is useful in the modeling.
#Moreover, some staff don't work for Trinity anymore, what is the point
#of knowing which staff can affect student decisions?.
#Consider removing this variable in the modeling stage.

#Column17 - Legacy
sum(is.na(clean_data$Legacy))#13658 NAs.
table(clean_data$Legacy)#No questionable category.
#Impute NAs with "No Legacy"
clean_data$Legacy[is.na(clean_data$Legacy)] <- "No Legacy"
#Legacy has many options, leading some options to having 
#only a small number of cases.
#I will group all the options into 3 categories so that each category
#has the chance to affect the response.
clean_data$Legacy <- ifelse(clean_data$Legacy == "Legacy", "Legacy", 
                            ifelse(clean_data$Legacy == "No Legacy", "No Legacy",
                                   ifelse(grepl("Legacy, Opt Out",clean_data$Legacy), 
                                          "Legacy, Opt Out", "Legacy")))
clean_data$Legacy <- as.factor(clean_data$Legacy)

#Column18 - Athlete
sum(is.na(clean_data$Athlete))#13120 NAs.
table(clean_data$Athlete)#No questionable category.
#Impute NAs with "Non-Athlete"
clean_data$Athlete[is.na(clean_data$Athlete)] <- "Non-Athlete"
#Similar to Column17, Column18 has many categories with a few cases.
#Group all options into three categories: 
#Athlete, Non-Athlete, and Athlete, Opt Out.
clean_data$Athlete <- ifelse(clean_data$Athlete == "Athlete", "Athlete", 
                             ifelse(clean_data$Athlete == "Non-Athlete", "Non-Athlete",
                                    ifelse(grepl("Opt Out",clean_data$Athlete), 
                                           "Athlete, Opt Out", "Athlete")))
clean_data$Athlete <- as.factor(clean_data$Athlete)                                                                                                                                                                                               

#Column19 - Sport.1.Sport
sum(is.na(clean_data$Sport.1.Sport))#13120 NAs.
table(clean_data$Sport.1.Sport)#No questionable category.
#Impute NAs with "No Sport".
clean_data$Sport.1.Sport[is.na(clean_data$Sport.1.Sport)] <- "No Sport"
#Group sport men and sport women into one group
#so that each group has sufficient cases to have an impact on the response.
clean_data$Sport.1.Sport <- ifelse(clean_data$Sport.1.Sport == "Baseball", "Baseball", 
                                   ifelse(clean_data$Sport.1.Sport == "Softball", "Softball",
                                          ifelse(clean_data$Sport.1.Sport == "Football", "Football", 
                                                 ifelse(clean_data$Sport.1.Sport == "No Sport", "No Sport", 
                                                        ifelse(grepl("Basketball", clean_data$Sport.1.Sport), "Basketball",
                                                               ifelse(grepl("Cross Country", clean_data$Sport.1.Sport), "Cross Country",
                                                                      ifelse(grepl("Diving", clean_data$Sport.1.Sport), "Diving",
                                                                             ifelse(grepl("Golf", clean_data$Sport.1.Sport), "Golf",
                                                                                    ifelse(grepl("Soccer", clean_data$Sport.1.Sport), "Soccer",
                                                                                           ifelse(grepl("Swimming", clean_data$Sport.1.Sport), "Swimming",
                                                                                                  ifelse(grepl("Tennis", clean_data$Sport.1.Sport), "Tennis",
                                                                                                         ifelse(grepl("Track", clean_data$Sport.1.Sport), "Track", "Volleyball"))))))))))))
clean_data$Sport.1.Sport <- as.factor(clean_data$Sport.1.Sport)

#Column20 - Sport.1.Rating
sum(is.na(clean_data$Sport.1.Rating))#13120 NAs.
table(clean_data$Sport.1.Rating)#No questionable category.
#Impute NAs with "No Sport".
clean_data$Sport.1.Rating[is.na(clean_data$Sport.1.Rating)] <- "No Sport"
clean_data$Sport.1.Rating<- factor(clean_data$Sport.1.Rating, order = TRUE, 
                                   levels = c("No Sport", "Varsity", "Blue Chip", "Franchise"))

#Column21 - Sport.2.Sport
sum(is.na(clean_data$Sport.2.Sport))#14513 NAs.
table(clean_data$Sport.2.Sport)#No questionable category.
#impute NAs with "No 2ndSport".
clean_data$Sport.2.Sport[is.na(clean_data$Sport.2.Sport)] <- "No 2ndSport"
#The number of cases for each sport type is very small (< about 1% of the data set).
#It's better to group all options into 2 categories: 2ndSport vs. No 2ndSport.
clean_data$Sport.2.Sport <- ifelse(clean_data$Sport.2.Sport == "No 2ndSport", 
                                   "No 2ndSport", "2ndSport")
clean_data$Sport.2.Sport <- as.factor(clean_data$Sport.2.Sport)

#Column22 - Sport.2.Rating
sum(is.na(clean_data$Sport.2.Rating))#15085 NAs.
table(clean_data$Sport.2.Rating)#No questionable category.
#Only 58 out of 15143 observations are rated. 
#This is less than 0.5% of the data set!
#I don't think Sport.2.Rating will have much impact on the response.
#Consider removing Column22 in the modeling stage.

#Column23 - Sport.3.Sport
sum(is.na(clean_data$Sport.3.Sport))#14907 NAs.
table(clean_data$Sport.3.Sport)#No questionable category.
#impute NAs with "No 3rdSport".
clean_data$Sport.3.Sport[is.na(clean_data$Sport.3.Sport)] <- "No 3rdSport"
#The number of cases for each sport type is very small (< 0.5% of the data set).
#It's better to group all options into 2 categories: 3rdSport vs. No 3rdSport.
clean_data$Sport.3.Sport <- ifelse(clean_data$Sport.3.Sport == "No 3rdSport", 
                                   "No 3rdSport", "3rdSport")
clean_data$Sport.3.Sport <- as.factor(clean_data$Sport.3.Sport)

#Column24 - Sport.3.Rating
sum(is.na(clean_data$Sport.3.Rating))#15140 NAs.
table(clean_data$Sport.3.Rating)#No questionable category.
#Only 3 out of 15143 observations are rated!
#Consider removing Column24 in the modeling stage.

#Column25 - Academic.Interest.1
sum(is.na(clean_data$Academic.Interest.1))#6 NAs.
table(clean_data$Academic.Interest.1)#No questionable category.
clean_data[is.na(clean_data$Academic.Interest.1),]
#Most of the NAs for Academic.Interest.1 have a value for Academic.Interest.2
#We may assign the corresponding values in Academic.Interest.2 
#to NAs in Academic.Interest.1 if Academic.Interest.2 has a value.
clean_data$Academic.Interest.1 <- ifelse(is.na(clean_data$Academic.Interest.1) == TRUE, 
                                         clean_data$Academic.Interest.2, 
                                         clean_data$Academic.Interest.1)
#For the remaining NAs in Academic.Interest.1, assign Undecided.
clean_data$Academic.Interest.1[is.na(clean_data$Academic.Interest.1)] <- "Undecided"
#Group Business related options into "Business".
#Based on your understanding of the academic fields listed in this variable,
#you may also consider grouping other related options into
#a broad major, similar to what I did with Business. 
clean_data$Academic.Interest.1 <- ifelse(grepl("Business", clean_data$Academic.Interest.1), "Business",
                                         ifelse(clean_data$Academic.Interest.1 == "Finance", "Business",
                                                ifelse(clean_data$Academic.Interest.1 == "Entrepreneurship", "Business", 
                                                       clean_data$Academic.Interest.1)))
#Group options with a low number of cases (< 100 cases) into "Others".
frequencies <-data.frame(table(clean_data$Academic.Interest.1))
frequencies
clean_data$Academic.Interest.1.Frequency <- NA
for (i in 1:nrow(clean_data)){
  for(j in 1:nrow(frequencies)){
    if (clean_data$Academic.Interest.1[i] == frequencies$Var1[j])
    {clean_data$Academic.Interest.1.Frequency[i] <- frequencies$Freq[j]}}
}

for (i in 1:nrow(clean_data)){
  if (clean_data$Academic.Interest.1.Frequency[i] < 100)
  {clean_data$Academic.Interest.1[i] <- "Other"}else{
    clean_data$Academic.Interest.1[i]
  }
}
clean_data$Academic.Interest.1 <- as.factor(clean_data$Academic.Interest.1)
#Remember to drop Academic.Interest.1.Frequency in the modeling stage.


#Column26 - Academic.Interest.2
sum(is.na(clean_data$Academic.Interest.2))#159 NAs.
#Replace repeated academic interests with Undecided, 
#then make NAs Undecided
clean_data$Academic.Interest.2 <- ifelse(clean_data$Academic.Interest.2 == clean_data$Academic.Interest.1, 
                                         "Undecided", clean_data$Academic.Interest.2)
clean_data$Academic.Interest.2[is.na(clean_data$Academic.Interest.2)] <- "Undecided"
table(clean_data$Academic.Interest.2)#No questionable category.
#Group Business related options into "Business".
clean_data$Academic.Interest.2 <- ifelse(grepl("Business", clean_data$Academic.Interest.2), "Business",
                                         ifelse(clean_data$Academic.Interest.2 == "Finance", "Business",
                                                ifelse(clean_data$Academic.Interest.2 == "Entrepreneurship", "Business", 
                                                       clean_data$Academic.Interest.2)))
#Group options with a low number of cases (<100 cases) into "Others".
frequencies <-data.frame(table(clean_data$Academic.Interest.2))
frequencies
clean_data$Academic.Interest.2.Frequency <- NA
for (i in 1:nrow(clean_data)){
  for(j in 1:nrow(frequencies)){
    if (clean_data$Academic.Interest.2[i] == frequencies$Var1[j])
    {clean_data$Academic.Interest.2.Frequency[i] <- frequencies$Freq[j]}}
}

for (i in 1:nrow(clean_data)){
  if (clean_data$Academic.Interest.2.Frequency[i] < 100)
  {clean_data$Academic.Interest.2[i] <- "Other"}else{
    clean_data$Academic.Interest.2[i]
  }
}
clean_data$Academic.Interest.2 <- as.factor(clean_data$Academic.Interest.2)
#Remember to drop Academic.Interest.2.Frequency in the modeling stage.


#Column27 - First_Source.Origin.First.Source.Summary
sum(is.na(clean_data$First_Source.Origin.First.Source.Summary))#No NA.
table(clean_data$First_Source.Origin.First.Source.Summary)#No questionable category.
#Group options with a low number of cases (< 100) into "Other Sources".
frequencies <-data.frame(table(clean_data$First_Source.Origin.First.Source.Summary))
frequencies
clean_data$First_Source.Summary.Frequency <- NA
for (i in 1:nrow(clean_data)){
  for(j in 1:nrow(frequencies)){
    if (clean_data$First_Source.Origin.First.Source.Summary[i] == frequencies$Var1[j])
    {clean_data$First_Source.Summary.Frequency[i] <- frequencies$Freq[j]}}
}

for (i in 1:nrow(clean_data)){
  if (clean_data$First_Source.Summary.Frequency[i] < 100)
  {clean_data$First_Source.Origin.First.Source.Summary[i] <- "Other Sources"}else{
    clean_data$First_Source.Origin.First.Source.Summary[i]
  }
}
clean_data$First_Source.Origin.First.Source.Summary <- as.factor(clean_data$First_Source.Origin.First.Source.Summary)
#Remember to drop First_Source.Summary.Frequency in the modeling stage.


#Column28 - Total.Event.Participation
sum(is.na(clean_data$Total.Event.Participation))#No NA.
table(clean_data$Total.Event.Participation)#No questionable category.
#3, 4, 5 combined accounts for < 1% of the data set.
#Compared to the number of cases in 0, 1, and 2, the number of cases
#in 3, 4, and 5 won't be very useful in predicting the response.
#Factor the variable and group 3, 4, and 5 into "2 or more".
clean_data$Total.Event.Participation <- ifelse(clean_data$Total.Event.Participation > 2,
                                               2, clean_data$Total.Event.Participation)
#Convert int to char so that level name can be modified.
clean_data$Total.Event.Participation <- as.character(clean_data$Total.Event.Participation)
clean_data$Total.Event.Participation <- ifelse(clean_data$Total.Event.Participation == "2",
                                               "2 or more", clean_data$Total.Event.Participation)
clean_data$Total.Event.Participation <- as.factor(clean_data$Total.Event.Participation)

#Column29 - Count.of.Campus.Visits
sum(is.na(clean_data$Count.of.Campus.Visits))#No NA.
table(clean_data$Count.of.Campus.Visits)#No questionable category.
#Factor the variable and group 5, 6, and 8 into 4.
clean_data$Count.of.Campus.Visits <- ifelse(clean_data$Count.of.Campus.Visits > 4,
                                            4, clean_data$Count.of.Campus.Visits)
#Convert int to char so that I can modify level name.
clean_data$Count.of.Campus.Visits <- as.character(clean_data$Count.of.Campus.Visits)
clean_data$Count.of.Campus.Visits <- ifelse(clean_data$Count.of.Campus.Visits == "4",
                                            "4 or more", clean_data$Count.of.Campus.Visits)
clean_data$Count.of.Campus.Visits <- as.factor(clean_data$Count.of.Campus.Visits)


#Column30 - School..1.Organization.Category
sum(is.na(clean_data$School..1.Organization.Category))#38 NAs.
table(clean_data$School..1.Organization.Category)#No questionable category.
#Only 16 cases belong to College but 15089 cases belong to High School.
#Should remove this variable.

#Column31 - School.1.Code
sum(is.na(clean_data$School.1.Code))#11879 NAs.
table(clean_data$School.1.Code)
#Will School Code matter much? Plus,there are 11879 missing values!
#Consider removing Column 31 in the modeling stage.

#Column32 - School.1.Class.Rank..Numeric.
sum(is.na(clean_data$School.1.Class.Rank..Numeric.))#8136 NAs.
#Column33 - School.1.Class.Size..Numeric.
sum(is.na(clean_data$School.1.Class.Size..Numeric.))#8136 NAs.
#Percentage rank can more accurately reflect a student's academic performance
#than numeric rank. 
#New Column - School.1.Top.Percent.in.Class
clean_data$School.1.Top.Percent.in.Class <- NA
clean_data$School.1.Top.Percent.in.Class <- 100 *(clean_data$School.1.Class.Rank..Numeric./clean_data$School.1.Class.Size..Numeric.)

sum(is.na(clean_data$School.1.Top.Percent.in.Class))
#Impute the 8136 NAs based on Academic.Index column. 
#Since I need to handle NAs in School.1.Top.Percent.in.Class
#according Academic.Index, first let's see whether Academic.Index needs to cleaned.
sum(is.na(clean_data$Academic.Index))#829 NAs.
table(clean_data$Academic.Index)#No questionable level.
#Impute 829 NAs with the most common level.
clean_data$Academic.Index[is.na(clean_data$Academic.Index)] <- 3
#No missing values in Academic.Index now.
#Impute missing values in School.1.Top.Percent.in.Class based on Academic.Index.
clean_index_1 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 1) %>%
  mutate(School.1.Top.Percent.in.Class = replace(School.1.Top.Percent.in.Class, 
                                                 is.na(School.1.Top.Percent.in.Class), mean(School.1.Top.Percent.in.Class, na.rm=TRUE)))

clean_index_2 <- clean_data %>% 
  group_by(Academic.Index) %>% 
  filter(Academic.Index == 2) %>%
  mutate(School.1.Top.Percent.in.Class = replace(School.1.Top.Percent.in.Class, 
                                                 is.na(School.1.Top.Percent.in.Class), mean(School.1.Top.Percent.in.Class, na.rm=TRUE)))

clean_index_3 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 3) %>%
  mutate(School.1.Top.Percent.in.Class = replace(School.1.Top.Percent.in.Class, 
                                                 is.na(School.1.Top.Percent.in.Class), mean(School.1.Top.Percent.in.Class, na.rm=TRUE)))  

clean_index_4 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 4) %>%
  mutate(School.1.Top.Percent.in.Class = replace(School.1.Top.Percent.in.Class, 
                                                 is.na(School.1.Top.Percent.in.Class), mean(School.1.Top.Percent.in.Class, na.rm=TRUE)))    

clean_index_5 <- clean_data %>% 
  group_by(Academic.Index) %>%
  filter(Academic.Index == 5) %>%
  mutate(School.1.Top.Percent.in.Class = replace(School.1.Top.Percent.in.Class, 
                                                 is.na(School.1.Top.Percent.in.Class), mean(School.1.Top.Percent.in.Class, na.rm=TRUE)))     

clean_data <- rbind(clean_index_1, clean_index_2, clean_index_3, clean_index_4, clean_index_5)
clean_data$Academic.Index <- as.numeric(clean_data$Academic.Index)

#Column34 - School.1.GPA
#Remove this variable in the modeling stage
#because School.1.GPA.Recalculated is more accurate.

#Column35 - School.1.GPA.Scale
#Remove this variable in the modeling stage as it is irrelevant.

#Column36 - School.1.GPA.Recalculated
sum(is.na(clean_data$School.1.GPA.Recalculated))#0 NA.
skewness(clean_data$School.1.GPA.Recalculated)
#Moderately skewed, consider transformation.
skewness(clean_data$School.1.GPA.Recalculated^10)
#-0.06391376 --> better
clean_data$School.1.GPA.Recalculated <- clean_data$School.1.GPA.Recalculated^10


#Column37 - School.2.Class.Rank..Numeric.
sum(is.na(clean_data$School.2.Class.Rank..Numeric.))#15143 NAs.
#All cases are blank. Remove this variable in the modeling stage.

#Column38 - School.2.Class.Size..Numeric.
sum(is.na(clean_data$School.2.Class.Size..Numeric.))#15143 NAs.
#All cases are blank. Remove this variable in the modeling stage.

#Column39 - School.2.GPA
sum(is.na(clean_data$School.2.GPA))#15143 NAs.
#All cases are blank. Remove this variable in the modeling stage.

#Column40 - School.2.GPA.Scale
sum(is.na(clean_data$School.2.GPA.Scale))#15143 NAs.
#All cases are blank. Remove this variable in the modeling stage.

#Column41 - School.2.GPA.Recalculated
sum(is.na(clean_data$School.2.GPA.Recalculated))#15143 NAs.
#All cases are blank. Remove this variable in the modeling stage.

#Column42 - School.3.Class.Rank..Numeric.
sum(is.na(clean_data$School.3.Class.Rank..Numeric.))#15143 NAs.
#All cases are blank. Remove this variable in the modeling stage.

#Column43 - School.3.Class.Size..Numeric.
sum(is.na(clean_data$School.3.Class.Size..Numeric.))#15143 NAs.
#All cases are blank. Remove this variable in the modeling stage.

#Column44 - School.3.GPA
sum(is.na(clean_data$School.3.GPA))#15143 NAs.
#All cases are blank. Remove this variable in the modeling stage.

#Column45 - School.3.GPA.Scale
sum(is.na(clean_data$School.3.GPA.Scale))#15143 NAs.
#All cases are blank. Remove this variable in the modeling stage.

#Column46 - School.3.GPA.Recalculated
sum(is.na(clean_data$School.3.GPA.Recalculated))#15143 NAs.
#All cases are blank. Remove this variable in the modeling stage.

#Column 47 - ACT.Composite
sum(is.na(clean_data$ACT.Composite))
#7502 NA values - this is most likely because students did not take both
#No changes here. Already imputed into TestScore.Adjusted

#Columns 48-52 - These are breakdowns by section. May or May not be useful in the future. For now, I will impute NAs with the mean. I will use these
sum(is.na(clean_data$ACT.English)) # 7883 - More NA values than for ACT.Composite.
#clean_data$ACT.English[is.na(clean_data$ACT.English)] <- mean(clean_data$ACT.English, na.rm = TRUE)

sum(is.na(clean_data$ACT.Reading)) # 7883 - More NA values than for ACT.Composite.
#clean_data$ACT.English[is.na(clean_data$ACT.Reading)] <- mean(clean_data$ACT.Reading, na.rm = TRUE)

sum(is.na(clean_data$ACT.Math)) # 7883 - More NA values than for ACT.Composite.
#clean_data$ACT.English[is.na(clean_data$ACT.Math)] <- mean(clean_data$ACT.Math, na.rm = TRUE)

sum(is.na(clean_data$ACT.Science.Reasoning)) # 7883 - More NA values than for ACT.Composite.
#clean_data$ACT.English[is.na(clean_data$ACT.Science.Reasoning)] <- mean(clean_data$ACT.Science.Reasoning, na.rm = TRUE)

sum(is.na(clean_data$ACT.Writing)) #  14886 - More NA values than for ACT.Composite. Expected, because writing section is not useful in ACT. Not many
#people take it.
#clean_data$ACT.English[is.na(clean_data$ACT.Writing)] <- mean(clean_data$ACT.Writing, na.rm = TRUE)

#Column 53 - SAT.I C + RM
sum(is.na(clean_data$SAT.I.CR...M)) # 14569 NA Values. This is fine. We just need to check if there is a recentered score already.
#If not, then convert.

#Column 54 - SAT R Evidence-Based Reading and Writing Section + Math Section
sum(is.na(clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section...Math.Section)) #6711 NA values

#Column 55 - Permanent Geomarket
sum(is.na(clean_data$Permanent.Geomarket))# 1 NA value
(summary(as.factor(clean_data$Permanent.Geomarket))) #Impute NA into other 
clean_data$Permanent.Geomarket <- ifelse(is.na(clean_data$Permanent.Geomarket), "(Other)", clean_data$Permanent.Geomarket)
#looks fine otherwise

#For decision tree, I have to limit levels to 32. This has 349+, so I will group everything past top 31 together
# For decision tree, I have to limit levels to 32. This has 349+, so I will group everything past top 31 together
freq_table <- table(clean_data$Permanent.Geomarket)
top_levels <- names(sort(freq_table, decreasing = TRUE)[1:31])
levels(clean_data$Permanent.Geomarket) <- c(levels(clean_data$Permanent.Geomarket), "Other")
clean_data$Permanent.Geomarket[!clean_data$Permanent.Geomarket %in% top_levels] <- "Other"
clean_data$Permanent.Geomarket <- as.factor(clean_data$Permanent.Geomarket)


#Column 56 - Citizenship Status
sum(is.na(clean_data$Citizenship.Status))# 0 NA value
(summary(as.factor(clean_data$Citizenship.Status))) #Looks fine
clean_data$Citizenship.Status <- as.factor(clean_data$Citizenship.Status)


#Column 57 - Academic Index
sum(is.na(clean_data$Academic.Index))# 0 NA value
(summary(as.factor(clean_data$Academic.Index))) #Looks fine
clean_data$Academic.Index<-as.factor(clean_data$Academic.Index)

#Column 58 - Intend to Apply for Financial Aid 
sum(is.na(clean_data$Intend.to.Apply.for.Financial.Aid.))# 21 NA value 
(summary(as.factor(clean_data$Intend.to.Apply.for.Financial.Aid.))) #Will replace NA with 0 
clean_data$Intend.to.Apply.for.Financial.Aid. <- ifelse(is.na(clean_data$Intend.to.Apply.for.Financial.Aid.), 0, clean_data$Intend.to.Apply.for.Financial.Aid.)

#Column 59 - Merit Award
sum(is.na(clean_data$Merit.Award))# 0 NA value 
(summary(as.factor(clean_data$Merit.Award))) #Will replace NA with 0 
#Some merit codes do not exist in the excel, but exist in the dataset. These are mostly variations of how much money they get. For example, "I" means
#internationl. Then 12.5 would mean 12.5k a year. 
#Instead of cleaning this column excessively, I will later create a merit amount column. This will say how much money they get per year. 


#New Column - Total Merit 
clean_data$TotalMerit = ifelse(clean_data$Merit.Award == "YO" | clean_data$Merit.Award == "XO" | clean_data$Merit.Award == "SEM" | clean_data$Merit.Award == "Z0" | clean_data$Merit.Award == "TTS", 47392, clean_data$Merit.Award)
for (i in seq_along(clean_data$TotalMerit)) {
  # Check if the value contains any letters
  if (grepl("[A-Za-z]", clean_data$TotalMerit[i])) {
    # Strip the letters and multiply by 1000
    clean_data$TotalMerit[i] <- as.numeric(gsub("[^0-9.]", "", clean_data$TotalMerit[i])) * 1000
  }
}
sum(is.na(clean_data$TotalMerit)) #0 NA values. Makes this much easier to work with 
clean_data$TotalMerit <- as.numeric(clean_data$TotalMerit)

#Column 60 - SAT concordance score 
#proper SAT score conversion for old scores
sum(is.na(clean_data$SAT.Concordance.Score..of.SAT.R.)) #6711
#Will adjust for TestScore.Adjusted later

#Column 61 - Act concordance score
sum(is.na(clean_data$ACT.Concordance.Score..of.SAT.R.)) # 8222
#Will clean further later

#Column 62 - ACT.Concordance.Score..of.SAT.
sum(is.na(clean_data$ACT.Concordance.Score..of.SAT.))  #15120 
#Will clean further later

#Column 63 - Test Optional
sum(is.na(clean_data$Test.Optional)) # 11901
summary(as.factor(clean_data$Test.Optional)) #There are many NA values. Most likely NA values represent 0. 
clean_data$Test.Optional[is.na(clean_data$Test.Optional)] <- 0


#Column 64-66 - Old SAT section breakdowns 
sum(is.na(clean_data$SAT.I.Math)) #14569
sum(is.na(clean_data$SAT.I.Writing)) #14573
sum(is.na(clean_data$SAT.I.Critical.Reading)) #14569
#Many NA values. Will do further cleaning later if I to use this column

#Column 67 - SAT R evidence based reading and writing section
sum(is.na(clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section)) # 8332
#Will clean later 

#Column 68 - SAT R Math Section
sum(is.na(clean_data$SAT.R.Math.Section)) #8332
#WIll clean later

#Column 69 - Decision
sum(is.na(clean_data$Decision)) #0 NA as expected. This is our Dependent variable, NA values would not make sense here
clean_data$Decision <- as.factor(clean_data$Decision)

#Column 70 - Submit First Source
sum(is.na(clean_data$Submit_FirstSource)) #0 NA Values
summary(as.factor(clean_data$Submit_FirstSource))
mean(clean_data$Submit_FirstSource) #Not sure how to treat
max((clean_data$Submit_FirstSource)) #Max 293?
min((clean_data$Submit_FirstSource)) #-48
#NO idea how to interpret - will leave be for now

#Column 71 - Submit Inquiry
sum(is.na(clean_data$Submit_Inquiry)) #0 NA values
summary(as.factor(clean_data$Submit_Inquiry)) #Again do not know how to interpret

#Column 72 - Academic Interest Frequency 1
sum(is.na(clean_data$Academic.Interest.1.Frequency)) #0 NA 
clean_data$Academic.Interest.1.Frequency<-as.numeric(clean_data$Academic.Interest.1.Frequency) #May not need this column - This column has weird data - will re-evaluate later

#Column 73 - Academic Interest Frequency 2
sum(is.na(clean_data$Academic.Interest.2.Frequency)) #0 NA 
clean_data$Academic.Interest.1.Frequency<- as.numeric(clean_data$Academic.Interest.2.Frequency) #May not need this column - This column has weird data - will re-evaluate later


#Column 74 - First_Source.Summary.Frequency
sum(is.na(clean_data$First_Source.Summary.Frequency)) #0 
clean_data$Academic.Interest.1.Frequency<-as.numeric(clean_data$First_Source.Summary.Frequency) #May not need this column - This column has weird data - will re-evaluate later

#Column 75 - School 1 Top Percent in Class
sum(is.na(clean_data$School.1.Top.Percent.in.Class)) #0 
clean_data$School.1.Top.Percent.in.Class< - as.numeric(clean_data$School.1.Top.Percent.in.Class)


##################################################################################################
################# ACT TO SAT CONVERSIONS##########################################################
##################################################################################################
#There are multiple columns based on ACT and SAT values.
#I will create one column that has the students highest score. If there is no score
#due to decision to be test free, I will impute the mean score of ACT. 


#New Column - TestScore.Adjusted
#This column will combine all test scores and take the highest for each student 
clean_data$TestScore.Adjusted = clean_data$ACT.Composite
sum(is.na(clean_data$TestScore.Adjusted)) # 7502 to fill 
#This is the starting point. I will now go through each test score related column and impute values into TestScore.Adjusted




any(is.na(clean_data$ACT.Concordance.Score..of.SAT.R.) & !is.na(clean_data$SAT.Concordance.Score..of.SAT.R.)) #no NA values in SAT that arent in ACt - good


clean_data$TestScore.Adjusted <- ifelse(
  clean_data$TestScore.Adjusted == "NA" & clean_data$ACT.Concordance.Score..of.SAT.R. != "NA",
  clean_data$ACT.Concordance.Score..of.SAT.R.,
  clean_data$TestScore.Adjusted
)

clean_data$TestScore.Adjusted <- ifelse(
  clean_data$TestScore.Adjusted <=  clean_data$ACT.Concordance.Score..of.SAT.R. & !is.na(clean_data$ACT.Concordance.Score..of.SAT.R.),
  clean_data$ACT.Concordance.Score..of.SAT.R.,
  clean_data$TestScore.Adjusted
)

sum(is.na(clean_data$TestScore.Adjusted)) #Same Amount of Nas - However, if ACT.Concordance Value is higher, will use that score. 



SAT = c(1600, 1590, 1580, 1570, 1560, 1550, 1540, 1530, 1520, 1510, 1500, 1490, 1480, 1470, 1460, 1450, 1440, 1430, 1420, 1410,1400, 1390, 1380, 1370, 1360, 1350, 1340, 1330, 1320, 1310, 1300, 1290, 1280, 1270, 1260, 1250, 1240, 1230, 1220, 1210, 1200, 1190, 1180, 1170, 1160, 1150, 1140, 1130, 1120, 1110, 1100, 1090, 1080, 1070, 1060, 1050, 1040, 1030, 1020, 1010, 1000, 990, 980, 970, 960, 950, 940, 930, 920, 910, 900, 890, 880, 870, 860, 850, 840, 830, 820, 810, 800, 790, 780, 770,760, 750, 740, 730, 720, 710, 700, 690, 680, 670, 660, 650, 640, 630, 620, 610, 600, 590)
ACT =  c(36, 36, 36, 36, 35, 35, 35, 35, 34, 34, 34, 34, 33, 33, 33, 33, 32, 32, 32, 31, 31, 31, 30, 30, 30, 29, 29, 29, 28, 28, 28, 27, 27, 27, 27, 26, 26, 26, 25, 25, 25, 24, 24, 24, 24, 23, 23, 23, 22, 22, 22, 21, 21, 21, 21, 20, 20, 20, 19, 19, 19, 19, 18, 18, 18, 17, 17, 17, 17, 16, 16, 16, 16, 15, 15, 15, 15, 15, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 12, 12, 12, 12, 11, 11, 11, 11, 10, 10, 10, 9, 9, 9)
length(SAT)
length(ACT)


clean_data$ACTConvert <- clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section...Math.Section


for (i in 1:nrow(clean_data)){
  value = clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section...Math.Section[i]
  pos <- which(SAT == value)
  if (length(pos) == 0) {
    act_score <- NA # handle missing values
  } else {
    act_score <- ACT[pos]
  }
  clean_data$ACTConvert[i] <- act_score
}

clean_data$TestScore.Adjusted <- ifelse((!is.na(clean_data$ACTConvert) & !is.na(clean_data$TestScore.Adjusted)) & clean_data$TestScore.Adjusted <= clean_data$ACTConvert, clean_data$ACTConvert, clean_data$TestScore.Adjusted)
#impute converted SAT scores into test score column
clean_data$TestScore.Adjusted <- ifelse(!is.na(clean_data$ACTConvert) & is.na(clean_data$TestScore.Adjusted), clean_data$ACTConvert, clean_data$TestScore.Adjusted)
sum(is.na(clean_data$TestScore.Adjusted)) #still 1724 NA values? 



#Any students who did not send scores will be given the average based on their academic index. I will do this at the end with the rest of the NA values
Aca1 <- subset(clean_data, Academic.Index == 1 & !is.na(TestScore.Adjusted))
average_1 <- mean(Aca1$TestScore.Adjusted)
average_1 #33.07628
Aca2 <- subset(clean_data, Academic.Index == 2 & !is.na(TestScore.Adjusted))
average_2 <- mean(Aca2$TestScore.Adjusted)
average_2 #30.54879
Aca3 <- subset(clean_data, Academic.Index == 3 & !is.na(TestScore.Adjusted))
average_3 <- mean(Aca3$TestScore.Adjusted)
average_3 #29.58218
Aca4 <- subset(clean_data, Academic.Index == 4 & !is.na(TestScore.Adjusted))
average_4 <- mean(Aca4$TestScore.Adjusted)
average_4 #28.67491
Aca5 <- subset(clean_data, Academic.Index == 5 & !is.na(TestScore.Adjusted))
average_5 <- mean(Aca5$TestScore.Adjusted)
average_5 #26.69703


#It is a little weird how cloes the averages for 2,3, and 4 are - may re-evaluate later


#To recenter un-recentered scores, I took the average of unrecentered, and recentered scores and added the difference to un - recentered
#capping at 1600. I am doing this because some of these scores would help remove additional NA values from TestScore.Adjusted
mean(clean_data$SAT.I.CR...M, na.rm = TRUE) #1461.852
mean(clean_data$SAT.R.Evidence.Based.Reading.and.Writing.Section...Math.Section, na.rm = TRUE) #1371.309

clean_data$SAT.I.CR...M <- clean_data$SAT.I.CR...M - 90
clean_data$SAT.I.CR...M[clean_data$SAT.I.CR...M>1600] <- 1600


clean_data$SAT.I.CR.CONVERT <-clean_data$SAT.I.CR...M
for (i in 1:nrow(clean_data)){
  value = clean_data$SAT.I.CR...M[i]
  pos <- which(SAT == value)
  if (length(pos) == 0) {
    act_score <- NA # handle missing values
  } else {
    act_score <- ACT[pos]
  }
  clean_data$SAT.I.CR.CONVERT[i] <- act_score
}


clean_data$TestScore.Adjusted <- ifelse(
  is.na(clean_data$TestScore.Adjusted) & !is.na(clean_data$SAT.I.CR.CONVERT),
  clean_data$SAT.I.CR.CONVERT,
  clean_data$TestScore.Adjusted
)
sum(is.na(clean_data$TestScore.Adjusted)) 


clean_data$TestScore.Adjusted <- ifelse(
  is.na(clean_data$TestScore.Adjusted) & !is.na(clean_data$ACT.Concordance.Score..of.SAT.),
  clean_data$ACT.Concordance.Score..of.SAT.,
  clean_data$TestScore.Adjusted
)
sum(is.na(clean_data$TestScore.Adjusted)) #Still some NA values. It seems that these overlap with TestScore.Adjusted

#Finally, I will impute the mean for all NA values left in the dataset based on academic index
clean_data$TestScore.Adjusted[is.na(clean_data$TestScore.Adjusted) & clean_data$Academic.Index == 1] <- average_1
clean_data$TestScore.Adjusted[is.na(clean_data$TestScore.Adjusted) & clean_data$Academic.Index == 2] <- average_2
clean_data$TestScore.Adjusted[is.na(clean_data$TestScore.Adjusted) & clean_data$Academic.Index == 3] <- average_3
clean_data$TestScore.Adjusted[is.na(clean_data$TestScore.Adjusted) & clean_data$Academic.Index == 4] <- average_4
clean_data$TestScore.Adjusted[is.na(clean_data$TestScore.Adjusted) & clean_data$Academic.Index == 5] <- average_5

sum(is.na(clean_data$TestScore.Adjusted))
#Now TestScore.Adjusted has all proper values. 
#I will not be explicitly converting all section scores, because there are only 2 sections for the SAT, 4 for the ACT, and 3 for the old SAT
#It will be very difficult to work with.

#Since I am using TestScore.Adjusted, I will be removing all other test score related columns


#######################################################PREPARING FOR MODEL BUILDING###########################################################################
#removing columns
clean_data_minus_cols <- clean_data[, -which(colnames(clean_data) %in% c("ID", "train.test", "Admit.Type", "Permanent.Postal", "First_Source.Summary.Frequency",
                                                                         "School..1.Organization.Category", "School.1.Code", "School.1.GPA.Scale", "School.2.GPA",
                                                                         "School.2.GPA.Scale", "School.2.GPA.Recalculated", "School.2.Class.Rank..Numeric.", "School.2.Class.Size..Numeric.", 
                                                                         "School.3.Class.Rank..Numeric.", "School.3.Class.Size..Numeric.","School.3.GPA",
                                                                         "School.3.GPA.Scale", "School.3.GPA.Recalculated", "ACT.English", "ACT.Reading", "ACT.Math",
                                                                         "ACT.Science.Reasoning", "ACT.Writing", "SAT.I.Math", "SAT.I.Writing", "SAT.I.Critical.Reading", 
                                                                         "SAT.R.Math.Section", "ACT.Composite", "SAT.I.CR...M", 
                                                                         "SAT.R.Evidence.Based.Reading.and.Writing.Section...Math.Section", 
                                                                         "SAT.Concordance.Score..of.SAT.R.", "ACT.Concordance.Score..of.SAT.R.", 
                                                                         "ACT.Concordance.Score..of.SAT.", "SAT.R.Evidence.Based.Reading.and.Writing.Section", "ACTConvert","SATConvert","SAT.I.CR.CONVERT", 
                                                                         "Inquiry.Date","Staff.Assigned.Name", "Sport.2.Rating", "Sport.3.Rating", "School.1.Class.Rank..Numeric.",
                                                                         "School.1.Class.Size..Numeric.", "School.1.GPA", "Permanent.Country", "Merit.Award", "First_Source.Origin.First.Source.Date", "Submitted"))]

#First of all, I removed columns like ID and train.test, which are not relevant to the decision. Then I also removed many columns which are either
#irrelevant, like GPA scales, because there is already a rescaled GPA column and columns with mostly NAs or only one type of data, such as school 2 and 3 data
#Finally, I removed all test score related variables, since I created TestScore.Adjusted. This combines all test score variables and the others
#should NOT be considered 
#I also removed inquiry date after I made a year and day of year variable

#splitting dataset based on project specifications
Train <- clean_data_minus_cols[1:10001,]
Test <-clean_data_minus_cols[10002:15144,]

#I already wrote the code to factor these earlier, but was getting errors so put them here again
Train$Permanent.Geomarket <- as.factor(Train$Permanent.Geomarket)
Test$Permanent.Geomarket <- as.factor(Test$Permanent.Geomarket)
Train$Citizenship.Status <- as.factor(Train$Citizenship.Status)
Test$Citizenship.Status <- as.factor(Test$Citizenship.Status)

#######################################################MODEL BUILDING###########################################################################
#A.	Logistic regression 
Logistic_model <- glm(Decision ~ ., family = binomial(link = "logit"),
                      data = Train) 
summary(Logistic_model)

#using 0.5 as initial cutoff point
Logistic_prob_train <- predict(Logistic_model, type = "response", Train)
Logistic_prob_train
Logistic_pred_train <- ifelse(Logistic_prob_train >= 0.5, 1, 0)
Logistic_conting_train <- table(Logistic_pred_train, Train$Decision, 
                                dnn = c("Predicted", "Actual"))
Logistic_conting_train
#for given circumstances not bad model. Very good at predicting 0, but bad at predicting 1 properly
#however, this makes sense as it would be easier to predict if a college student will not accept than if they would accept since most students
#do not accept - they have many acceptances etc and have to choose 1.

Logistic_cm_train <- confusionMatrix(Logistic_conting_train)
Logistic_cm_train
#0.5286 

#running the test model with 0.5
Logistic_prob_test <- predict(Logistic_model, type = "response", Test)
Logistic_pred_test <- ifelse(Logistic_prob_test >= 0.5, 1, 0)
Logistic_conting_test <- table(Logistic_pred_test, Test$Decision, 
                               dnn = c("Predicted", "Actual"))
Logistic_cm_test <- confusionMatrix(Logistic_conting_test)
Logistic_cm_test
#0.5004

cutoffs = c(0.2, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.8)
kappa_score_test = c()
kappa_score_train = c()


#The code for my loop is not perfect, but it helps me compare all of the values easily

for (i in (1:10)) {
  Logistic_prob_train <- predict(Logistic_model, type = "response", Train)
  Logistic_pred_train <- ifelse(Logistic_prob_train >= cutoffs[i], 1, 0)
  Logistic_conting_train <- table(Logistic_pred_train, Train$Decision, 
                                  dnn = c("Predicted", "Actual"))
  Logistic_cm_train <- confusionMatrix(Logistic_conting_train)
  
  Logistic_prob_test <- predict(Logistic_model, type = "response", Test)
  Logistic_pred_test <- ifelse(Logistic_prob_test >= cutoffs[i], 1, 0)
  Logistic_conting_test <- table(Logistic_pred_test, Test$Decision, 
                                 dnn = c("Predicted", "Actual"))
  Logistic_cm_test <- confusionMatrix(Logistic_conting_test)
  
  print(paste("Kappa score (train):", Logistic_cm_train$overall['Kappa'], "Kappa score (test):", Logistic_cm_test$overall['Kappa'], "Cutoff value:", cutoffs[i]))
}
#[1] "Kappa score (train): 0.524899872827689 Kappa score (test): 0.451598963046177 Cutoff value: 0.2"
#[1] "Kappa score (train): 0.57267103102875 Kappa score (test): 0.49123409568234 Cutoff value: 0.3"
#[1] "Kappa score (train): 0.571412431432845 Kappa score (test): 0.503188955427641 Cutoff value: 0.35"
#[1] "Kappa score (train): 0.558461263187223 Kappa score (test): 0.512848225410867 Cutoff value: 0.4"
#[1] "Kappa score (train): 0.543923355770899 Kappa score (test): 0.513843345813695 Cutoff value: 0.45"
#[1] "Kappa score (train): 0.506961744823297 Kappa score (test): 0.501122251377327 Cutoff value: 0.55"
#[1] "Kappa score (train): 0.479204683323679 Kappa score (test): 0.477046580931316 Cutoff value: 0.6"
#[1] "Kappa score (train): 0.452612270404802 Kappa score (test): 0.461638513353726 Cutoff value: 0.65"
#[1] "Kappa score (train): 0.426038887327136 Kappa score (test): 0.434306653361655 Cutoff value: 0.7"
#[1] "Kappa score (train): 0.338526309004179 Kappa score (test): 0.386471596170836 Cutoff value: 0.8"

#A cutoff of 0.45 gives the highest test kappa at 0.513843345813695 . It also has a train kappa of 0.543923355770899, which although not the highest, is still
#good

#B.	KNN
#KNN only takes numerics
non_numeric_cols <- Train %>%
  select_if(~!is.numeric(.)) %>%
  names()
print(non_numeric_cols)

# Select categorical columns
cols <- c("Sex", "Ethnicity", "Race", "Religion", "Application.Source", "Decision.Plan", "Legacy", "Athlete", "Sport.1.Sport", "Sport.1.Rating", "Sport.2.Sport", "Sport.3.Sport", "Academic.Interest.1", "Academic.Interest.2", "First_Source.Origin.First.Source.Summary", "Total.Event.Participation", "Count.of.Campus.Visits", "Permanent.Geomarket", "Citizenship.Status")
cols <- intersect(colnames(clean_data), cols)
clean_data_numeric <- clean_data_minus_cols[, -which(colnames(clean_data_minus_cols) %in% cols)]
clean_data_numeric2 <- clean_data_numeric 
#only numeric data now - will need to re-add columns one at a time
clean_data_numeric$dummy_Sex <- ifelse(clean_data_minus_cols$Sex == "M", 1, 0)
dummies_ethnicity <- dummy_cols(clean_data_minus_cols$Ethnicity)
dummies_Religion <- dummy_cols(clean_data_minus_cols$Religion)
dummies_Application.Source <- dummy_cols(clean_data_minus_cols$Application.Source)
dummies_Decision.Plan<- dummy_cols(clean_data_minus_cols$Decision.Plan)
dummies_Legacy<-  recode(clean_data$Legacy, 
                         "Opt Out" = 0, 
                         "Non-Legacy" = 1, 
                         "Legacy" = 2)

dummies_Athlete<- dummy_cols(clean_data_minus_cols$Athlete)
dummies_Sport.1.Rating<- recode(clean_data$Sport.1.Rating, 
                                "No Sport" = 0, 
                                "Varsity" = 1, 
                                "Blue Chip" = 2,
                                "Franchise" = 3)


clean_data_numeric$Total.Event.Participation <- ifelse(clean_data_minus_cols$Total.Event.Participation == "2 or more", 2,
                                                       ifelse(clean_data_minus_cols$Total.Event.Participation == "1", 1, 0))

clean_data_numeric$Count.of.Campus.Visits <- ifelse(clean_data_minus_cols$Count.of.Campus.Visits == "4 or more", 4,
                                                    ifelse(clean_data_minus_cols$Count.of.Campus.Visits == "3", 3, 
                                                           ifelse(clean_data_minus_cols$Count.of.Campus.Visits == "2", 2, 
                                                                  ifelse(clean_data_minus_cols$Count.of.Campus.Visits == "1", 1, 0))))
summary(as.factor(clean_data_numeric$Count.of.Campus.Visits))
dummies_Citizenship.Status<- dummy_cols(clean_data_minus_cols$Citizenship.Status)

summary(as.factor(clean_data_minus_cols$Citizenship.Status))


clean_data_numeric <- cbind(clean_data_numeric, dummies_Citizenship.Status, dummies_Legacy, dummies_Sport.1.Rating, dummies_ethnicity, dummies_Religion, dummies_Application.Source, dummies_Decision.Plan, dummies_Athlete)


non_numeric_cols <- clean_data_numeric %>%
  select_if(~!is.numeric(.)) %>%
  names()
print(non_numeric_cols)
clean_data_numeric$Academic.Index <- as.numeric(clean_data_numeric$Academic.Index)
clean_data_numeric <- subset(clean_data_numeric, select = -c(28,32,34,38,54,59,66)) # removing columns I dummycolled 

#Other non-numerics will stay removed because they have 20+ columns which would make the model VERY slow
clean_data_numeric$Decision <- as.numeric(clean_data_numeric$Decision)
num_missing <- sum(is.na(clean_data_numeric))

# print result
print(num_missing)
Train_Numeric <- clean_data_numeric[1:10001,]
Test_Numeric <-clean_data_numeric[10002:15144,]

Kappa <- rep(0, 100)
for(i in 1:100){
  set.seed(1)
  nn_test <- kNN(Decision ~., train = Train_Numeric, test = Test_Numeric, k = i)
  nn_conting_test <- table(nn_test, Test$Decision, 
                           dnn = c("Predicted", "Actual"))
  nn_cm_test <- confusionMatrix(nn_conting_test)
  Kappa[i] <- nn_cm_test$overall["Kappa"]
}

which.max(Kappa)
#i=x has the highest Kappa score

NN_train <- kNN(Purchase ~., train = Train_Numeric, test = Train_Numeric, k = x)
NN_conting_train <- table(NN_train, Train_Numeric$Decision, 
                          dnn = c("Predicted", "Actual"))
NN_cm_train <- confusionMatrix(NN_conting_train)
NN_cm_train
Kappa_train_NN <- NN_cm_train$overall["Kappa"]
Kappa_train_NN

NN_test <- kNN(Purchase ~., train = Train_Numeric, test = Test_Numeric, k = x)
NN_conting_test <- table(NN_test, Test$Decision, 
                         dnn = c("Predicted", "Actual"))
NN_cm_test <- confusionMatrix(NN_conting_test)
Kappa_test_NN <- NN_cm_test$overall["Kappa"]
Kappa_test_NN

#C.	Simple classification tree

simple_tree <- tree(Decision ~., split = "gini", Train)
summary(simple_tree)


plot(simple_tree)
text(simple_tree, pretty = 0)
#too dense to get much out of it.

#Training
simple_tree_pred_train <- predict(simple_tree, Train, type = "class")
simple_tree_conting_train <- table(simple_tree_pred_train, 
                                   Train$Decision, 
                                   dnn = c("Predicted", "Actual"))
simple_tree_conting_train
simple_tree_cm_train <- confusionMatrix(simple_tree_conting_train)
simple_tree_cm_train
simple_tree_cm_train$overall["Kappa"]
#0.6813245 - high Kappa

#Test 
simple_tree_pred_test <- predict(simple_tree, Test, type = "class")
simple_tree_conting_test <- table(simple_tree_pred_test, 
                                  Test$Decision, 
                                  dnn = c("Predicted", "Actual"))
simple_tree_conting_test
simple_tree_cm_test <- confusionMatrix(simple_tree_conting_test)
simple_tree_cm_test
simple_tree_cm_test$overall["Kappa"]
#0.2763174 - Extremely overfitted it seems

#D.	Tree Pruning##############################################################################################################################

cv_Tree <- cv.tree(simple_tree, FUN = prune.misclass, K = 10)
cv_Tree$size[which.min(cv_Tree$dev)] #664
plot(cv_Tree$size, cv_Tree$dev, type = "b")

prune_tree <- prune.misclass(simple_tree, best = 664)
plot(prune_tree)
text(prune_tree, pretty = 0)

# Training
prune_tree_pred_train <- predict(prune_tree, Train, type = "class")
prune_tree_conting_train <- table(prune_tree_pred_train, 
                                  Train$Decision, 
                                  dnn = c("Predicted", "Actual"))
prune_tree_conting_train
prune_tree_cm_train <- confusionMatrix(prune_tree_conting_train)
prune_tree_cm_train
prune_tree_cm_train$overall["Kappa"]
#0.6787247

#Test
prune_tree_pred_test <- predict(prune_tree, Test, type = "class")
prune_tree_conting_test <- table(prune_tree_pred_test, 
                                 Test$Decision, 
                                 dnn = c("Predicted", "Actual"))
prune_tree_conting_test
prune_tree_cm_test <- confusionMatrix(prune_tree_conting_test)
prune_tree_cm_test
prune_tree_cm_test$overall["Kappa"]
#test 0.2777758 - still low

#E.	Classification trees with Bagging ##############################################################################################################################
Tree_Bagging <- randomForest(Decision ~ ., data = Train,
                             ntrees = 500, mtry = 42, split = "gini", 
                             replace = TRUE, importance = TRUE)

which.min(Tree_Bagging$err.rate[ , 1])
#optimal number of trees is 361

#Training
bag_train_pred <- predict(Tree_Bagging, Train, type = "class")
bag_conting_train <- table(bag_train_pred, Train$Decision, 
                           dnn = c("Predicted", "Actual"))
bag_conting_train
bag_cm_train <- confusionMatrix(bag_conting_train)
bag_cm_train
bag_cm_train$overall["Kappa"]
#training Kappa - 1

#Test
bag_test_pred <- predict(Tree_Bagging, Test, type = "class")
bag_conting_test <- table(bag_test_pred, Test$Decision, 
                          dnn = c("Predicted", "Actual"))
bag_conting_test
bag_cm_test <- confusionMatrix(bag_conting_test)
bag_cm_test
bag_cm_test$overall["Kappa"]
#Test - 0.4992877

#F.	Classification trees with Random Forests ##############################################################################################################################
Test_Kappa_RF <- rep(0, 42)
for(i in 1:42){
  set.seed(1)
  Tree_RF <- randomForest(Decision ~ ., data = Train,
                          ntrees = 500, mtry = i, split = "gini", replace = TRUE,
                          importance = TRUE)
  Test_pred_RF <- predict(Tree_RF, Test, type = "class")
  RF_conting_test <- table(Test_pred_RF, Test$Decision, 
                           dnn = c("Predicted", "Actual"))
  RF_cm_test <- confusionMatrix(RF_conting_test)
  Test_Kappa_RF[i] <- RF_cm_test$overall["Kappa"]
}
which.max(Test_Kappa_RF)
#mtry = 6 gives the highest Kappa.
Test_Kappa_RF[which.max(Test_Kappa_RF)]

Tree_RF <- randomForest(Decision ~ ., data = Train,
                        ntrees = 500, mtry = 6, replace = TRUE,
                        importance = TRUE)
rf_train_pred <- predict(Tree_RF, Train, type = "class")
rf_conting_train <- table(rf_train_pred, Train$Decision, 
                          dnn = c("Predicted", "Actual"))
rf_conting_train
rf_cm_train <- confusionMatrix(rf_conting_train)
rf_cm_train$overall["Kappa"]
#Kappa is 1? - most likely super overfitted

rf_test_pred <- predict(Tree_RF, Test, type = "class")
rf_conting_test <- table(rf_test_pred, Test$Decision, 
                         dnn = c("Predicted", "Actual"))
rf_conting_test
rf_cm_test <- confusionMatrix(rf_conting_test)
rf_cm_test$overall["Kappa"]
#Kappa is 0.5250855


#G.	Classification trees with boosting ##############################################################################################################################

n_trees <- rep(0, 6)
min_cv_error <- rep(0, 6)
for(i in 1:6){
  set.seed(1)
  Tree_Boosting <- gbm(Decision ~., data = Train, distribution = "bernoulli",
                       n.trees = 500, interaction.depth = i, cv.folds = 10,
                       shrinkage = 0.01)
  n_trees[i] <- which.min(Tree_Boosting$cv.error)
  min_cv_error[i] <- Tree_Boosting$cv.error[which.min(Tree_Boosting$cv.error)]
}



library(gbm)
for(i in 1:6){
  set.seed(1)
  tryCatch({
    Tree_Boosting <- gbm(Decision ~., data = Train, distribution = "bernoulli",
                         n.trees = 500, interaction.depth = i, cv.folds = 10,
                         shrinkage = 0.01)
    n_trees[i] <- which.min(Tree_Boosting$cv.error)
    min_cv_error[i] <- Tree_Boosting$cv.error[which.min(Tree_Boosting$cv.error)]
  }, error = function(e) {
    print(paste0("Error in loop ", i, ": ", e$message))
  })
}


which.min(min_cv_error)
#minimum is 4
n_trees[4]
#402 trees

set.seed(1)
Tree_Boosting <- gbm(Decision ~., data = Train, 
                     distribution = "bernoulli",
                     n.trees = 402 , interaction.depth = 4,
                     shrinkage = 0.01)
summary(Tree_Boosting)


cutoffs = c(0.2, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65, 0.7, 0.8)
kappa_score_test = c()
kappa_score_train = c()


#The code for my loop is not perfect, but it helps me compare all of the values easily

library(caret)
for (i in (1:10)) {
  boost_prob_train <- predict(Tree_Boosting, type = "response", 
                              Train)
  boost_pred_results_train <- ifelse(boost_prob_train > cutoffs[i], 1, 0)
  boost_conting_train <- table(boost_pred_results_train, Train$Decision, 
                               dnn = c("Predicted", "Actual"))
  boost_cm_train <- confusionMatrix(boost_conting_train)
  
  
  boost_prob_test <- predict(Tree_Boosting, type = "response", Test)
  boost_pred_results_test <- ifelse(boost_prob_test > cutoffs[i], 1, 0)
  boost_conting_test <- table(boost_pred_results_test, Test$Decision, 
                              dnn = c("Predicted", "Actual"))
  boost_cm_test <- confusionMatrix(boost_conting_test)
  
  print(paste("Kappa score (train):", boost_cm_train$overall["Kappa"], "Kappa score (test):", boost_cm_test$overall["Kappa"], "Cutoff value:", cutoffs[i]))
}


"Kappa score (train): 0.4691223961769 Kappa score (test): 0.485860005027691 Cutoff value: 0.2"
"Kappa score (train): 0.536400993923701 Kappa score (test): 0.511076474898872 Cutoff value: 0.3"
"Kappa score (train): 0.540387252267366 Kappa score (test): 0.501159708282233 Cutoff value: 0.35"
"Kappa score (train): 0.525451060666346 Kappa score (test): 0.48973207747559 Cutoff value: 0.4"
"Kappa score (train): 0.480437630729572 Kappa score (test): 0.461191271218017 Cutoff value: 0.45"
"Kappa score (train): 0.359971094635363 Kappa score (test): 0.358172588146845 Cutoff value: 0.55"
"Kappa score (train): 0.323226893588267 Kappa score (test): 0.315324152572401 Cutoff value: 0.6"
"Kappa score (train): 0.266131869849482 Kappa score (test): 0.283803386163206 Cutoff value: 0.65"
"Kappa score (train): 0.217002381153054 Kappa score (test): 0.247182949695999 Cutoff value: 0.7"
"Kappa score (train): 0.133156756582581 Kappa score (test): 0.174730307025243 Cutoff value: 0.8"
#The best test Kappa was 0.511076474898872 for a cutoff value of 0.3. The train Kappa was 
#0.536400993923701




#H.	Support vector machine with Linear kernel  ##############################################################################################################################
svc_best_cost <- tune(METHOD = svm, train.x = Decision ~., train.y = NULL,
                      data = Train, kernel = "linear", 
                      ranges = list(cost = 2^(-5:8)))

#Output the best cost parameter
svc_best_cost$best.parameters
#
svc_best_cost$best.performance
#

best_svm_linear <- svc_best_cost$best.model
best_svm_linear

#best coefficients for linear kernel ##############################################################################################################################
coefficients_best_svm_linear <- t(best_svm_linear$SV) %*% best_svm_linear$coefs
coefficients_best_svm_linear
coeff_names <- rownames(coefficients_best_svm_linear)
coeff_abs <- abs(coefficients_best_svm_linear)
coeff_data <- data.frame(coeff_names, coeff_abs, row.names = NULL)
coeff_data[with(coeff_data, order(-coeff_abs)),]  


svc_pred_train_best <- predict(best_svm_linear, Train)
svc_conting_train_best <- table(svc_pred_train_best, Train$Purchase, 
                                dnn = c("Predicted", "Actual"))
svc_conting_train_best
svc_confu_train_best <- confusionMatrix(svc_conting_train_best)
svc_confu_train_best$overall["Kappa"]
#Train Kappa

svc_pred_test_best <- predict(best_svm_linear, Test)
svc_conting_test_best <- table(svc_pred_test_best, Test$Purchase, 
                               dnn = c("Predicted", "Actual"))
svc_conting_test_best
svc_confu_test_best <- confusionMatrix(svc_conting_test_best)
svc_confu_test_best$overall["Kappa"]
#Test Kappa 


#For now I will only use a coarse search so that my code will run in time
#

#I.	Support vector machine with Polynomial kernel
set.seed(1)
library(e1071)
svmp_best_coarse <- tune.svm(x = Decision ~., kernel = "polynomial",
                             data = Train, cost = 2^(-5:8), 
                             degree = 2:10)

best_svmp_coarse <- svmp_best_coarse$best.model
best_svmp_coarse

#Most important variables
coeff_best_svmp <- t(best_svmp_coarse$SV) %*% best_svmp_coarse$coefs
coeff_best_svmp
coeff_names <- rownames(coeff_best_svmp)
coeff_abs <- abs(coeff_best_svmp)
coeff_data <- data.frame(coeff_names, coeff_abs, row.names = NULL)
coeff_data[with(coeff_data, order(-coeff_abs)), ] 

svmp_pred_train_best <- predict(best_svmp_coarse, Train)
svmp_conting_train_best <- table(svmp_pred_train_best, Train$Purchase, 
                                 dnn = c("Predicted", "Actual"))
svmp_conting_train_best
svmp_confu_train_best <- confusionMatrix(svmp_conting_train_best)
svmp_confu_train_best$overall["Kappa"]
#

svmp_pred_test_best <- predict(best_svmp_coarse, Test)
svmp_conting_test_best <- table(svmp_pred_test_best, Test$Purchase, 
                                dnn = c("Predicted", "Actual"))
svmp_conting_test_best
svmp_confu_test_best <- confusionMatrix(svmp_conting_test_best)
svmp_confu_test_best$overall["Kappa"]
#


#J.	Support vector machine with Radial kernel ##############################################################################################################################
svmr_best_coarse <- tune.svm(x = Decision ~., kernel = "radial",
                             data = Train, cost = 2^(seq(-5:8)), 
                             gamma = 2^(seq(-15:8)))

svmr_best_fine$best.performance 
#
svmr_best_fine$best.parameters 
#

best_svmr_fine <-svmr_best_fine$best.model

svmr_pred_train_best_fine <- predict(best_svmr_fine, Train)
svmr_conting_train_best_fine <- table(svmr_pred_train_best_fine,
                                      Train$Purchase, 
                                      dnn = c("Predicted", "Actual"))

svmr_conting_train_best_fine
svmr_confu_train_best_fine <- confusionMatrix(svmr_conting_train_best_fine)
svmr_confu_train_best_fine$overall["Kappa"]
#Train Kappa

svmr_pred_test_best_fine <- predict(best_svmr_fine, Test)
svmr_conting_test_best_fine <- table(svmr_pred_test_best_fine,
                                     Test$Purchase, 
                                     dnn = c("Predicted", "Actual"))

svmr_conting_test_best_fine
svmr_confu_test_best_fine <- confusionMatrix(svmr_conting_test_best_fine)
svmr_confu_test_best_fine$overall["Kappa"]
#Test Kappa
