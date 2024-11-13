#### ___________Freshness sensor algorithm_______________#########
  ### Analyizing sensor resopnse and 
    ### correlating them with the TVBN data
    ### Author: Sadra (Mohammad) Avestan

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import datetime
from pylab import rcParams

from sadra import pubfig as pf
pf.setup(label_font=14, tick_font=14, axis_width=2, tick_major_width=3,
         tick_minor_width=2.5, tick_major_size=4, tick_minor_size=3, showminorticks=True)

path_dir = '/home/sadra/Desktop/kroger/tentamus/tentamus_jan_2023/Sensor_data_as_of_12_29_2022'
plot_dir = '/home/sadra/Desktop/kroger/tentamus/tentamus_jan_2023/plots'
## STEAK FRIDGE CONTANERS
A = ['IC000152.csv', 'IC000183.csv','IC000195.csv', 'RT000151.csv','IC000186.csv','IC000153.csv']##
B = ['RT000161.csv','IC000148.csv','RT000160.csv','IC000194.csv','RT000190.csv'] ## ,'IC000143.csv'
C = ['IC000188.csv','IC000160.csv','RT000163.csv','IC000168.csv','RT000182.csv','RT000170.csv']
D = ['RT000167.csv','IC000169.csv','RT000157.csv','RT000181.csv','IC000182.csv','RT000154.csv'] #
E = ['RT000177.csv','RT000156.csv','RT000162.csv','IC000174.csv','RT000150.csv'] #all the res values are zero, 'IC000144.csv'

frdg_container_list = {'A':A, 'B':B, 'C':C, 'D':D, 'E':E}

tvbn_average_frdg = [26.856532,28.68096,33.92424,31.4081946534653,29.12105,32.38102,34.86462,44.41542,45.89046,43.21842,48.44707,75.70206]
tvbn_days_frdg    = [2.000,3.000 ,5.000 ,9.000 ,11.000,12.000,13.000,16.000,17.000,18.000,19.000,20.000]

########################################### CHANGE THE COLUMN NAME########################
## list of CSV files in the directory
column_name=['unknown','IC','Date_Time','Temp','Res']
refridgerated = os.listdir(path_dir + "/refridgerated")
ambient = os.listdir(path_dir + '/ambient')
csv_files = [('refridgerated',refridgerated), ('ambient',ambient)] # make a list of lists

# ##Loop through lists
# for tmp_test, files in csv_files:
#     for file in files:
#         # Print the file path
#         print(file)
#         print(path_dir +"/%s/%s"%(tmp_test, file))
#         df = pd.read_csv(path_dir +"/%s/%s"%(tmp_test, file))

#         # if df.columns.all()==None:  ## check if the file already has column name
#         if df.columns.tolist() != column_name:
#             df.columns=column_name ## add the column name

#         else: print('column already has a name')

#         # creat new directory or check if it already exsit
#         if not os.path.exists(path_dir+'/all_data_label'):
#             os.mkdir(path_dir+'/all_data_label')
#             print("directory created")

#         else:print('directory already exisits')
#         df.to_csv(path_dir+"/all_data_label/"+file, index=False) # save the file
#         df = pd.read_csv(path_dir+"/all_data_label/"+file)
       


#         ##################Temporal_adjustment#################
#         if 'Time_Min' in df.columns and 'Time_day' in df.columns:
#             print('The columns already exist')

#         else:
#             df['Date_Time'] = pd.to_datetime(df['Date_Time'])
#             min_time = df.Date_Time.min() ## minimu Time in the column
#             df['Time_Min'] = (df.Date_Time - min_time).dt.total_seconds() / 60
#             df['Time_day'] = np.round(df.Time_Min/1440, 3) # round it to 5 digits
#             df.to_csv(path_dir+"/all_data_label/"+file, index=False) # save the file

##########################################################################################        

### PLOT RAW DATA

### starting point for taking the data of the fridge is 11:15 am of 12/5
###            which in data is index number of 2041. in case some dat are missing
###             i get the index every time.

rcParams['figure.figsize']=[10,6]
############_________READ DATA_____________#######################
def read_data(file, data_type):

    df = pd.read_csv(path_dir+"/all_data_label/"+file)
    if data_type=='raw_data':
        init_indx = 0
        last_indx = len(df.Res)
        # print('last_index')
        # print('it is raw data')
    else:
        init_indx = df.loc[df["Time_day"] > 2.830].index[0] ## STARTING DAY OF MEASURING MEAT RESPONSE
        # indx = 2041
        last_indx=df[df['Time_day'] > 20.000].index[0]
        # print('last_indx: ', last_indx, 'init_indx: ', init_indx)
   
    return df, init_indx, last_indx

#################___REMOVE 0 VALUES__________###########################
def replace_zeros_with_mean10(df):
        ###____FInd first NON_zero value afetr the current zero
    # df['Res'] = df["Res"].replace(0, method='bfill')
        ##find the first 10 non-zero values,
            ##   calculate the mean of those values,
            #  and replace the current zero value with that mean value.
    i = 0
    while i < len(df['Res']):
        if df["Res"][i] == 0:
            non_zero_values = []
            for j in range(i+1, len(df['Res'])):
                if df['Res'][j] != 0:
                    non_zero_values.append(df["Res"][j])
                if len(non_zero_values) == 10:
                    break
            mean_value = np.mean(non_zero_values)
            df.at[i, 'Res'] = mean_value
        i += 1
    # print('df replace 0s:  ',df)

    return df
#########################################################################
###___________Replace large value with the average of 10 values before_____

def replace_large_values_with_mean10(df):
    rolling_window = 50
    threshold = 100 # set a threshold for the spikes
    for j in range(rolling_window, len(df.Res)):   # iterate through the data points
        rolling_avg = np.mean(df.Res[j-rolling_window:j]/1000) ## rol_avg of the previous 10 data pt/1000
       
        if abs(df.Res[j]/1000 - rolling_avg)> threshold:
            df.loc[j, 'Res'] = rolling_avg*1000
    # print("j: ", j, 'rolling_avg: ', rolling_avg, df_data1.Res[j]/1000)
    # break
    # print('df.Res after large value replacement: ', df['Res'])
    return df

#####################  CLEAN THE OUTLIERS #################################
def clean_outlier(file, data_type):

    df_data, init_indx, last_indx = read_data(file, data_type)
    # print('first len: ', len(df_data.Res), df_data.Res.head(5))
    df_data1 = df_data[init_indx:last_indx].copy()
    df_data1 =df_data1.reset_index(drop=True)
    # print('seconde len: ', len(df_data1.Res), df_data1.head(5))
   
    df_0 = replace_zeros_with_mean10(df_data1)
    df = replace_large_values_with_mean10(df_0)

    df['Time_day'] =  df['Time_day'].sub(0.830)  ### ADJUST WITH THE MEAT MEASURING STARTIN TIME
   
    return df, init_indx, last_indx
#######################__________AVERAGE OF EACH CONTAINER________##############
def find_longest_df(cntnr, data_type, cntnr_label):
    dataframes = []
    for i, file in enumerate(cntnr):
        df, init_indx, last_indx = clean_outlier(file, data_type)

        # Change column names to unique names
        df.columns = [f"{i}_{col}" for col in df.columns]
        dataframes.append(df)

    # print('dataframes', dataframes)
    longest_index = max(range(len(dataframes)), key=lambda i: len(dataframes[i]['%s_Time_day'%i]))
    longest_df = dataframes[longest_index]

    # merge all dataframes on "Time_day" column and fill missing values with the matched values from the longest dataframe
    merged_df = pd.concat(dataframes, axis=1, join='outer')

    # calculate the average of "Res" column
    for col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna(method='ffill')    

    Res_mean = merged_df.filter(like="Res").mean(axis=1)/1000
    longest_df_time = longest_df[longest_df.columns[longest_df.columns.str.contains("Time_day")]]
 
    df_Res_Time = pd.concat([Res_mean, longest_df_time], axis=1)
    df_Res_Time.columns = ['Res', 'Time_day']

    return Res_mean, longest_df_time, df_Res_Time  

#################____AVERAGE OF ALL THE CONTAINERS______################################
def Res_mean_all_cntnrs(df_Res_Time_all):

    longest_index = max(range(len(df_Res_Time_all)), key=lambda i: len(df_Res_Time_all[i]['Time_day']))
    longest_df = df_Res_Time_all[longest_index]
    # merge all dataframes on "Time_day" column and fill missing values with the matched values from the longest dataframe
    merged_df = pd.concat(df_Res_Time_all, axis=1, join='outer')
    # calculate the average of "Res" column
    for col in merged_df.columns:
        merged_df[col] = merged_df[col].fillna(method='ffill')    
        print('col: ', col)
    print('merged_df head: ', merged_df.head(5))
    # print('merged_df tail: ', merged_df['0_Res'].tail(5))
    Res_mean = merged_df.filter(like="Res").mean(axis=1)
    longest_df_time = longest_df[longest_df.columns[longest_df.columns.str.contains("Time_day")]]

    df_Res_Time_all = pd.concat([Res_mean, longest_df_time], axis=1)
    df_Res_Time_all.columns = ['Res_all_cntnr', 'Time_day']
    print('df_Res_Time_all.head(): ',df_Res_Time_all.head())
    return  df_Res_Time_all

########________R_in_TVBN_time_point___________#############
def R_in_TVBN_time_point(df,tvbn_days_frdg, tim_col, R_col):
    result = []
    for i in tvbn_days_frdg:
        time_day = min(df['Time_day'], key=lambda x: abs(x-i))
        print('time_day: ', time_day)
        res = df.loc[df['Time_day'] == time_day, 'Res_all_cntnr'].values[0]
        result.append(res)

    print(result)
    return result
##################################################################################################
# ####################### PLOT DATA ###############################
def plot_frdg_steak(cntnrs, title_label,tvbn_value, tvbn_day, data_type):
   
    fig, axs = plt.subplots(3,3)
    Res_avrg_all_cntnr = []

    for i, (cntnr_label, cntnr) in enumerate(cntnrs.items()):
        x=i//3
        y=i%3

        for file in cntnr:
            
            if data_type == 'raw_data':
                df, init_indx, last_indx = read_data(file, data_type)
                axs[1,2].set_axis_off()
                axs[2,y].set_axis_off()
                axs[x,y].plot(df.Time_day[init_indx:last_indx], df.Res[init_indx:last_indx]/1000, alpha=.6, label="%s"%file)
                # axs[x,0].set_ylabel('R (k\u03A9)')
                # axs[1,y].set_xlabel('Time (days)')

            else:
                # ###########_______CLEAN OUTLIERS_________#############
                df, init_indx, last_indx = clean_outlier(file, data_type)  
                # print('new Time_day: \n', df.head(5))
                axs[x,y].plot(df.Time_day, df.Res/1000, alpha=.6, label="%s"%file)  
                axs[x,0].set_ylabel('R (k\u03A9)')
                axs[2,y].set_xlabel('Time (days)')
        ###################################################
                # curve_sum.append(df.Res/1000)
                # print('file: ', file, 'len(df.R): ',len(df.Res), 'cntnr_label: ', cntnr_label)
                # print('head: ', df.head(5))
                # print('tail: ', df.tail(5))

            axs[x,0].set_ylabel('R (k\u03A9)')
            axs[1,y].set_xlabel('Time (days)')
            axs[x,y].set_ylim(0,800)
            axs[x,y].set_title('container_%s'%cntnr_label)
            axs[x,y].legend(loc='upper right', fontsize = 5)
            plt.subplots_adjust(wspace=0.5, hspace=0.6)
        ###########_____FIND THE LONGEST FILE AND ADJUST THE REST OF THE FILE________###############
                        ########____AND TAKE THE AVERAGE OF EACH CONTAINER__________###############
        Res_mean, longest_df_time, df_Res_Time   = find_longest_df(cntnr, data_type, cntnr_label)
        axs[x,y].plot(longest_df_time, Res_mean, alpha=1, label="Avrg_R", c='black')  
        axs[x,y].legend(loc='upper right', fontsize = 5)
        # print(cntnr_label, ": len: ", len(Res_mean), "\n", 'Res_mean: ', Res_mean.head(4))
        # print('longest_df_time: ', longest_df_time.head())
        # break
        ############_____AVERAGE OF ALL THE CONTAINORS___________##############
        Res_avrg_all_cntnr.append(df_Res_Time)
        # print('Res_avrg_all_cntnr_head: ', Res_avrg_all_cntnr.head(4), 'Res_avrg_all_cntnr_tail: ', Res_avrg_all_cntnr.tail(4) )
    df_Res_Time_all = Res_mean_all_cntnrs(Res_avrg_all_cntnr)
    axs[1,2].plot(df_Res_Time_all['Time_day'], df_Res_Time_all['Res_all_cntnr'], label='all_cntnrs_avrg', c='m')
    axs[1,2].set_title('All_Cntnrs_Avrg')
    axs[1,2].set_ylim(0,800)
    # axs[1,2].legend(loc='upper right', fontsize = 5)
        #########################################################
    #####_____________PLOT AVERAGE TVBN________###############
    axs[2,0].plot(tvbn_days_frdg ,tvbn_average_frdg,  'o-',c='g')
    axs[2,0].set_xlabel('Time (days)')
    axs[2,0].set_ylabel('Avrg_TVBN')
    axs[2,0].set_title("TVBN vs Time")
    ################_POINTS OF RESISTANCE IN TVBN MEASURED TIME__#########################
    avrg_R_in_TVBN_time =  R_in_TVBN_time_point(df_Res_Time_all,tvbn_days_frdg, 'Time_day','Avrg_R')
    axs[2,1].plot(tvbn_days_frdg ,avrg_R_in_TVBN_time,  'o-',c='r')
    axs[2,1].set_xlabel('Time (days)')
    axs[2,1].set_ylabel('Avrg_R_all')
    axs[2,1].set_title("R for the reported TVBN")
    ####______TVBN vs R________####################
    axs[2,2].plot(tvbn_average_frdg ,avrg_R_in_TVBN_time,  'o-',c='plum')
    axs[2,2].set_xlabel('Avrg_TVBN')
    axs[2,2].set_ylabel('Avrg_R_all')
    axs[2,2].set_title("R vs TVBN")
    ######################################################
    img_name = plot_dir+'/%s_%s.png'%(title_label, data_type)  
    plt.savefig(img_name,bbox_inches='tight', dpi=300)
    plt.close()
       
# plot_frdg_steak(frdg_container_list, 'steak4deg',tvbn_average_frdg, tvbn_days_frdg, 'raw_data')
plot_frdg_steak(frdg_container_list, 'steak4deg',tvbn_average_frdg, tvbn_days_frdg, 'clean_data')

# def remove_spikes():
