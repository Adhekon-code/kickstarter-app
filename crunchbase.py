#Selecting an age range 
# Only include companies that are 3 to 7 years

def filtered_data(df):
    df = df[(df["age"] >= 3*365) &(df["age"] >= 7*365)]

    df.loc[:, "age"] = df["age"] - 3*365


