# Loading requisite packages
library(rvest)
library(dplyr)
library(xml2)

# Reading statcast and war data
statcast = read.csv("/Users/Jake/Desktop/STA 221/Project Data/statcast.csv", header = TRUE)
war = read.csv("/Users/Jake/Desktop/STA 221/Project Data/WAR.csv", header = TRUE)

# Stripping invisible characters from each dataset Name variable
war$Name = gsub("[[:cntrl:]]", "", war$Name)
war$Name = gsub("\u00A0", " ", war$Name)
statcast$last_name..first_name = gsub("\u00A0", " ", statcast$last_name..first_name)
statcast$last_name..first_name = gsub("[[:cntrl:]]", "", statcast$last_name..first_name)

# Removing * and # from the Name variable in the war dataset
war$Name = gsub("\\*", "", war$Name)
war$Name = gsub("\\#", "", war$Name)

# Editing the statcast name variable to match the war name column
names(statcast)[1] = "Name"
statcast$Name = sub("(.*), (.*)", "\\2 \\1", statcast$Name)

# Removing special characters and Jr./III from names in both datasets
statcast$Name = gsub("í", "i", statcast$Name)
statcast$Name = gsub("é", "e", statcast$Name)
statcast$Name = gsub("ú", "u", statcast$Name)
statcast$Name = gsub("á", "a", statcast$Name)
statcast$Name = gsub(" Jr.", "", statcast$Name)
statcast$Name = gsub(" III", "", statcast$Name)
statcast$Name = gsub(" II", "", statcast$Name)
war$Name = gsub("í", "i", war$Name)
war$Name = gsub("é", "e", war$Name)
war$Name = gsub("ú", "u", war$Name)
war$Name = gsub("á", "a", war$Name)
war$Name = gsub(" Jr.", "", war$Name)
war$Name = gsub(" III", "", war$Name)
war$Name = gsub(" II", "", war$Name)

# Adjusting specific names within the statcast dataset to match the war dataset
statcast$Name[statcast$Name == "Cal Mitchell"] = "Calvin Mitchell"
statcast$Name[statcast$Name == "Dae-Ho Lee"] = "Dae-ho Lee"
statcast$Name[statcast$Name == "Diego A. Castillo"] = "Diego Castillo"
statcast$Name[statcast$Name == "Ji Man Choi"] = "Jiman Choi"
statcast$Name[statcast$Name == "Jose A. Martinez"] = "Jose Martinez"
statcast$Name[statcast$Name == "Michael Morse"] = "Mike Morse"
statcast$Name[statcast$Name == "Norichika Aoki"] = "Nori Aoki"
statcast$Name[statcast$Name == "Phillip Ervin"] = "Phil Ervin"

# Removing rows from war dataset that contain players with same names as players in statcast data
exclude = c("youngch03", "roberda09", "castidi01", "ramirjo02", "garcilu03", "garcilu05", "duffyma02", "smithwi04")
war = war[!(war$Name.additional %in% exclude), ]

# Merging the war data with the statcast data
names(statcast)[3] = "Year"
war = war[, c("Name", "Year", "WAR", "Name.additional")]
statcast = statcast[, c("Name", "Year", "player_id", names(statcast)[4:length(names(statcast))])]
statcast = merge(statcast, war, by = c("Name", "Year"), all.x = TRUE)

# Removing any rows in the statcast dataset corresponding to 2019, since the following season WAR will correspond to COVID-shortened 2020 season
statcast = statcast[statcast$Year != 2019, ]

# Scraping the career average WAR per season for each player-season row in the statcast dataset
avg.war = c()
next.war = c()
for (r in 1:length(statcast$Name)){
  
  # Storing the current year and player codename
  yr = statcast$Year[r]
  nm = statcast$Name.additional[r]
  
  # Defining the url for baseball-reference to visit and reading its html content
  url = paste("https://www.baseball-reference.com/players/", substr(nm, 1, 1), "/", nm, ".shtml", sep = "")
  page = read_html(url)

  # Extracting the commented HTML section containing the "Player Value -- Batting" table
  commented_section = page %>%
    html_nodes(xpath = "//comment()") %>%
    html_text() %>%
    paste(collapse = "")

  # Parsing the commented section as HTML
  commented_html = read_html(commented_section)

  # Creating the player value table for this player
  batting_value_table = commented_html %>%
    html_node("#batting_value") %>%
    html_table(fill = TRUE)

  # Calculating the player's average WAR per plate appearance for their career through the current year
  end_row = which(batting_value_table$Year == yr)[1]
  avg_WAR = sum(as.numeric(batting_value_table$WAR[1:end_row]), na.rm = TRUE)/sum(as.numeric(batting_value_table$PA[1:end_row]), na.rm = TRUE)

  # Appending this avg_WAR value to the avg.war vector
  avg.war = c(avg.war, avg_WAR)
  
  # If player does not have next season WAR, setting nw to NA, if they do have next season WAR, taking sum of all WAR values in following season
  if (!(yr + 1) %in% batting_value_table$Year){
    nw = NA
  } else {
    nw = sum(as.numeric(batting_value_table$WAR[batting_value_table$Year == (yr + 1)]), na.rm = TRUE)
  }
  
  # Appending the next season WAR value for each player to the next.war vector
  next.war = c(next.war, nw)
  
  # Resting for 3 seconds to avoid too many requests per minute
  Sys.sleep(3)
}

# Appending the avg.war and next.war variable to the statcast dataset
statcast$avg.war = avg.war
statcast$next.war = next.war

# Scraping injury data from first page of injuries at prosportstransactions.com
url = "https://www.prosportstransactions.com/baseball/Search/SearchResults.php?Player=&Team=&BeginDate=2014-01-01&EndDate=2024-12-31&InjuriesChkBx=yes&submit=Search"
page = read_html(url)
injuries = page %>%
  html_node("table") %>%
  html_table(fill = TRUE)

names(injuries) = injuries[1, ]
injuries = injuries[-1, ]


# Defining a for loop that will loop over all pages of the injury data
for (i in seq(25, 12300, 25)){
  url = paste("https://www.prosportstransactions.com/baseball/Search/SearchResults.php?Player=&Team=&BeginDate=2014-01-01&EndDate=2024-12-31&InjuriesChkBx=yes&submit=Search",
              "&start=", i, sep = "")
  page = read_html(url)
  table = page %>%
    html_node("table") %>%
    html_table(fill = TRUE)
  names(table) = table[1, ]
  table = table[-1, ]
  injuries = rbind(injuries, table)
}

# Creating the final injuries dataset based on rows that contain "surgery" or "out indefinitely" in the notes
injuries2 = injuries[grepl("surgery|out indefinitely", injuries$Notes, ignore.case = TRUE) &
                       !grepl("Tommy John", injuries$Notes, ignore.case = TRUE), ]

# cleaning the Relinquished column to get a consistent naming convention with the statcast dataset
injuries2$Relinquished = sub("^• ", "", injuries2$Relinquished)
injuries2$Relinquished = gsub("[[:cntrl:]]", "", injuries2$Relinquished)
injuries2$Relinquished = sub(" \\(.*$", "", injuries2$Relinquished)
injuries2$Relinquished = gsub(" Jr.", "", injuries2$Relinquished)
injuries2$Relinquished = sub("^.* / ", "", injuries2$Relinquished)
injuries2$Relinquished[128] = "Randal Grichuk"
injuries2$Relinquished[149] = "Garrett Mitchell"

# Removing the Acquired column and renaming the Relinquished Column as Name
injuries2 = injuries2[, -3]
names(injuries2)[3] = "Name"

# Subsetting the injuries dataframe to only players who are in the statcast data
injuries_subset = injuries2[injuries2$Name %in% statcast$Name, ]

# Converting the injury date variable to just the year; converting Date/Year variable to numeric
injuries_subset$Date = substr(injuries_subset$Date, 1, 4)
injuries_subset$Date = as.numeric(injuries_subset$Date)
statcast$Year = as.numeric(statcast$Year)

# Creating a inj variable in statcast, setting all equal to 0 (no injury)
statcast$inj = rep(0, length(statcast$Name))

# Iterating over all entries in the statcast dataframe, changing any player-year combos to 1 if injury in p12m
for (i in 1:length(statcast$Year)){
  if (statcast$Name[i] %in% injuries_subset$Name){
    player_sub = injuries_subset[injuries_subset$Name == statcast$Name[i], ]
    if ((statcast$Year[i] - 1) %in% player_sub$Date){
      statcast$inj[i] = 1
    }
  }
}

# Writing the final statcast_final dataset
#write.csv(statcast, file = "/Users/Jake/Desktop/STA 221/Project Data/statcast_final.csv", row.names = FALSE)

