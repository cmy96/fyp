import kaplan_meier as KM

filters_dict = {}
filters_dict["age_lower"] = 30
filters_dict["age_upper"] = 100
filters_dict["Race"] = "chinese"
filters_dict["T"] = "t2"
filters_dict["N"] = "n0"
filters_dict["M"] = "m0"
filters_dict["ER"] = "positive"
filters_dict["PR"] = "positive"
filters_dict["Her2"] = "negative"

# Load the Main DF
listToDrop = ['NRIC','dob','Has Bills?','Side','Hospital','KKH','NCCS','SGH','END_OF_ENTRY']
input_df = KM.kaplan_meier_load_clinical_df(listToDrop)

# output_df = KM.generate_kaplan_meier_with_filters(filters_dict = filters_dict, input_df=input_df, survival_type="DFS")
# print(output_df)

os,dfs,css = KM.generate_kaplan_meier_with_filters_for_all_survival_types(filters_dict,input_df)

# print(os.head())

