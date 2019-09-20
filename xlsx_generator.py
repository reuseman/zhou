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

    # Swap here tp and tn for the old bugged results
    tp = int(result[0])
    fp = int(result[1])
    fn = int(result[2])
    tn = int(result[3])

    return tp, fp, fn, tn


def read_original_zhou_results(csv_path):
    results = dict()
    with open(csv_path, mode="r") as results_file:
        csv_reader = csv.reader(results_file, delimiter=",")
        for row in csv_reader:
            if csv_reader.line_num == 1:
                continue
            results[row[0]] = [row[1], row[2], row[3], row[4]]

    return results


def read_metrics_zhou_results(results, record):
    tp, tn, fp, fn = 0, 0, 0, 0

    tp = int(results[record][0])
    tn = int(results[record][1])
    fp = int(results[record][2])
    fn = int(results[record][3])

    se, sp, ppv, acc = utils.compute_metrics(tp, tn, fp, fn)
    mcc = utils.compute_mcc(tp, tn, fp, fn)

    return [record, "ZHOU", tp, tn, fp, fn, se, sp, ppv, acc, mcc]


# Sum two arrays with the same length. Result in first array
def sum_array(array1, array2):
    for i in range(len(array1)):
        array1[i] += array2[i]


# Settings
n_algorithms = 8    # with zhou too

# Paths
main_dir_path = Path().absolute()
datasets_path = Path.joinpath(main_dir_path, "dataset")
generated_path = Path.joinpath(datasets_path, "generated")
generated_explicit_path = Path.joinpath(generated_path, "explicit_entropy")
result_explict_xlsx = Path.joinpath(generated_path, "explicit_entropy.xlsx")
result_original_zhou_path = Path.joinpath(datasets_path, "afdb_result_Bc.csv")

# The results of the original zhou algorithm are read

folders = [x.stem for x in generated_explicit_path.iterdir()]

algorithms_results = []
original_results = read_original_zhou_results(result_original_zhou_path)

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
            # Metrics in the following two lines will be overwritten below,
            #   hence they are useless
            se, sp, ppv, acc = utils.compute_metrics(tp, tn, fp, fn)
            mcc = utils.compute_mcc(tp, tn, fp, fn)
            value = [l1o, alg, tp, tn, fp, fn, se, sp, ppv, acc, mcc]
            algorithms_results.append(value)

    algorithms_results.append(read_metrics_zhou_results(original_results, l1o))

counter = 0
black_line_format = workbook.add_format()
black_line_format.set_bg_color("black")

great = workbook.add_format()
great.set_bold()
great.set_bg_color("#e8f2a1")

normal = workbook.add_format()

row = 1
zhou_dist = n_algorithms - 1  # in this case 6 - 1 = 5
for i in range(1, len(algorithms_results) + 1):
    for col in range(len(value)):
        apply_format = False
        # If TP/TN are bigger than zhou's ones or FP/FN are smaller format great
        # just one line can be used to print values though, without the if block
        # worksheet.write(row, col, algorithms_results[i - 1][col]
        not_a_string = not(isinstance(algorithms_results[i - 1][col], str) or isinstance(algorithms_results[i - 1 + zhou_dist][col], str))
        tp_or_tn_bigger_than_zhou = (
            not_a_string and
            col in [2, 3, 6, 7, 8, 9, 10]
            and algorithms_results[i - 1][col]
            > algorithms_results[i - 1 + zhou_dist][col]
        )
        fp_or_fn_smaller_than_zhou = (
            not_a_string and
            (col == 4 or col == 5)
            and algorithms_results[i - 1][col]
            < algorithms_results[i - 1 + zhou_dist][col]
        )

        if zhou_dist != 0 and (tp_or_tn_bigger_than_zhou or fp_or_fn_smaller_than_zhou):
            worksheet.write(row, col, algorithms_results[i - 1][col], great)
            current_format = great
        else:
            worksheet.write(row, col, algorithms_results[i - 1][col])
            current_format = normal

        if col == 6:
            formula = "=C{0}/(C{0}+F{0})".format(row + 1)
            worksheet.write_formula(row, col, formula, current_format)
        elif col == 7:
            formula = "=D{0}/(D{0}+E{0})".format(row + 1)
            worksheet.write_formula(row, col, formula, current_format)
        elif col == 8:
            formula = "=C{0}/(C{0}+E{0})".format(row + 1)
            worksheet.write_formula(row, col, formula, current_format)
        elif col == 9:
            formula = "=(C{0}+D{0}) / (C{0}+D{0}+E{0}+F{0})".format(row + 1)
            worksheet.write_formula(row, col, formula, current_format)
        elif col == 10:
            num = "(C{0}*D{0} - E{0}*F{0})".format(row + 1)
            den = "((C{0}+E{0})*(C{0}+F{0})*(D{0}+E{0})*(D{0}+F{0}))".format(row + 1)
            formula = "={}/sqrt({})".format(num, den)
            worksheet.write_formula(row, col, formula, current_format)

    # A black line to divide the different records is added
    if algorithms_results[i - 1][1] == "ZHOU":
        row += 1
        worksheet.set_row(row, 5, black_line_format)
        zhou_dist = n_algorithms
    row += 1
    zhou_dist -= 1

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
