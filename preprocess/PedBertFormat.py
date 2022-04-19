import csv

import pandas as pd

from common.common import save_obj

MAX_LEN = 30
SEP_TOKEN = 'SEP'


def values2codes(col, offset=0):
    dictionaray = dict((j, i + offset) for i, j in enumerate(col.unique()))
    return col.replace(dictionaray), dictionaray


def add_patient_age(event_df, patient_df, id_name='Id', birth_date_name='BIRTHDATE', event_date_name='START'):
    patient_short_df = patient_df[[id_name, birth_date_name]]

    output_df = pd.merge(event_df, patient_short_df, left_on='PATIENT', right_on=id_name)

    output_df[event_date_name] = pd.to_datetime(output_df[event_date_name])
    output_df[birth_date_name] = pd.to_datetime(output_df[birth_date_name])
    output_df['age'] = (output_df[event_date_name] - output_df[birth_date_name]).dt.days
    return output_df.drop([id_name, birth_date_name], axis=1)


def build_matrix(rows, cols, default=None):
    matrix = []
    for r in range(0, rows):
        matrix.append([default for c in range(0, cols)])
    return matrix


def save_matrix(path, data):
    with open(path, "w+", newline='') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(data)


def delete_empty_rows(input_data):
    output_data = []
    for row in input_data:
        if len(list(filter(None, row))) > 0:
            output_data.append(row)
        else:
            print("line deleted")
    return output_data


def main():
    # Load files
    patients_df = pd.read_csv("../data/patients.csv")
    conditions_df = pd.read_csv("../data/conditions.csv")
    medications_df = pd.read_csv('../data/medications.csv')

    # Filter unnecessary columns
    patients_df = patients_df[['Id', 'BIRTHDATE', 'DEATHDATE', 'RACE', 'ETHNICITY', 'GENDER', ]]
    conditions_df = conditions_df[['PATIENT', 'START', 'STOP', 'CODE']]
    medications_df = medications_df[['PATIENT', 'START', 'STOP', 'CODE']]

    # Calculate age
    conditions_df = add_patient_age(conditions_df, patients_df)
    # medications_df = add_patient_age(medications_df, patients_df)

    # Assign codes to string values
    patients_df['GENDER'], gender_dict = values2codes(patients_df['GENDER'])
    patients_df['ETHNICITY'], ethnicity_dict = values2codes(patients_df['ETHNICITY'])
    patients_df['RACE'], race_dict = values2codes(patients_df['RACE'])

    temp, condition_dict = values2codes(conditions_df['CODE'], offset=5)
    # medications_df['CODE'], medication_dict = values2codes(medications_df['CODE'])

    # Sort
    conditions_df = conditions_df.sort_values(by=['PATIENT', 'START'])
    medications_df = medications_df.sort_values(by=['PATIENT', 'START'])

    # Generate code and age sequence
    patients_list = patients_df['Id'].values
    code = build_matrix(len(patients_list), MAX_LEN)  # np.zeros(shape=(len(patients_list), MAX_LEN)) - 99
    age = build_matrix(len(patients_list), MAX_LEN)  # np.zeros(shape=(len(patients_list), MAX_LEN)) - 99

    # Format Data in BERT configuration
    patient_index = 0
    for patient in patients_list:
        tmp_df = conditions_df[conditions_df['PATIENT'] == patient].reset_index(drop=True)
        if tmp_df.size > 0:
            token_index = 1
            previous_age = tmp_df.loc[0, 'age']
            for index, row in tmp_df.iterrows():
                if row['age'] != previous_age:
                    code[patient_index][token_index] = SEP_TOKEN
                    age[patient_index][token_index] = str(round(previous_age / 365))
                    token_index += 1

                code[patient_index][token_index] = str(row['CODE'])
                age[patient_index][token_index] = str(round(row['age'] / 365))
                token_index += 1
                previous_age = row['age']
            patient_index += 1

    # Store Data in CSV
    save_matrix("../data/codes.csv", delete_empty_rows(code))
    save_matrix("../data/ages.csv", delete_empty_rows(age))

    # Store code2idx  dictionary
    condition_dict["PAD"] = 0
    condition_dict["UNK"] = 1
    condition_dict["CLS"] = 2
    condition_dict["SEP"] = 3
    condition_dict["MASK"] = 4

    save_obj({'token2idx': condition_dict}, "../data/dict")

    print("Program Terminated!")


if __name__ == "__main__":
    main()
