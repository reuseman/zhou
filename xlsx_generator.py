from pathlib import Path
import csv
import xlsxwriter
import utils


def read_results(csv_path):
    tp, tn, fp, fn = 0, 0, 0, 0

    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        result = next(csv_reader)

    tp = int(result[0])
    fp = int(result[1])
    fn = int(result[2])
    tn = int(result[3])

    return tp, fp, fn, tn


# Sum two arrays with the same length. Result in first array
def sum_array(array1, array2):
    for i in range(len(array1)):
        array1[i] += array2[i]


# Paths
main_dir_path = Path().absolute()
datasets_path = Path.joinpath(main_dir_path, "dataset")
generated_path = Path.joinpath(datasets_path, "generated")
generated_explicit_path = Path.joinpath(generated_path, "explicit_entropy")
result_explict_xlsx = Path.joinpath(generated_path, "explicit_entropy.xlsx")

folders = [x.stem for x in generated_explicit_path.iterdir()]

algorithms_results = []

# Write .xlsx file
workbook = xlsxwriter.Workbook(result_explict_xlsx)
worksheet = workbook.add_worksheet()
worksheet.freeze_panes(1, 1)
worksheet.set_column("A:K", 20)
worksheet.set_row(0, 18)

header = [
    "L1O",
    "Algorithm",
    "TP",
    "TN",
    "FP",
    "FN",
    "Sensitivity",
    "Specificity",
    "Precision",
    "Accuracy",
    "MCC",
]

header_format = workbook.add_format(
    {
        "bold": True,
        "align": "center",
        "valign": "vcenter",
        "fg_color": "#D7E4BC",
        "border": 1,
    }
)

for col in range(len(header)):
    worksheet.write(0, col, header[col], header_format)

for folder in generated_explicit_path.iterdir():
    for csv_file in folder.iterdir():
        if csv_file.suffix == ".csv":
            l1o = folder.stem
            alg = csv_file.stem
            tp, fp, fn, tn = read_results(csv_file)
            print(l1o, alg, tp, fp, fn, tn)
            se, sp, ppv, acc = utils.compute_metrics(tp, tn, fp, fn)
            mcc = utils.compute_mcc(tp, tn, fp, fn)
            value = [l1o, alg, tp, tn, fp, fn, se, sp, ppv, acc, mcc]
            algorithms_results.append(value)


for row in range(1, len(algorithms_results) + 1):
    for col in range(len(value)):
        worksheet.write(row, col, algorithms_results[row - 1][col])
        if col == 6:
            formula = "=C{0}/(C{0}+F{0})".format(row + 1)
            worksheet.write_formula(row, col, formula)
        elif col == 7:
            formula = "=D{0}/(D{0}+E{0})".format(row + 1)
            worksheet.write_formula(row, col, formula)
        elif col == 8:
            formula = "=C{0}/(C{0}+E{0})".format(row + 1)
            worksheet.write_formula(row, col, formula)
        elif col == 9:
            formula = "=(C{0}+D{0}) / (C{0}+D{0}+E{0}+F{0})".format(row + 1)
            worksheet.write_formula(row, col, formula)
        elif col == 10:
            num = "(C{0}*D{0} - E{0}*F{0})".format(row + 1)
            den = "((C{0}+E{0})*(C{0}+F{0})*(D{0}+E{0})*(D{0}+F{0}))".format(row + 1)
            formula = "={}/sqrt({})".format(num, den)
            worksheet.write_formula(row, col, formula)

print(row)
""" for folder in generated_explicit_path.iterdir():
    for csv_file in folder.iterdir():
        if csv_file.suffix == ".csv":
            result = read_results(csv_file)
            if csv_file.stem in algorithms_results:
                sum_array(algorithms_results[csv_file.stem], result)
            else:
                algorithms_results[csv_file.stem] = result """

workbook.close()
