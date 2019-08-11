from pathlib import Path
import utils

main_dir_path = Path().absolute()
dataset_path = Path.joinpath(main_dir_path, "dataset")
object_file_name = Path().joinpath(dataset_path, "afdb_result")
csv_file_name = Path().joinpath(dataset_path, "afdb_result.csv")

results = utils.read_object(object_file_name)

tp, tn, fp, fn = utils.compute_classifications(results)
se, sp, ppv, acc = utils.compute_metrics(tp, tn, fp, fn)

print("-----------------------------------")
print("TRUE POSITIVE : ", tp)
print("TRUE NEGATIVE : ", tn)
print("FALSE POSITIVE: ", fp)
print("FALSE NEGATIVE: ", fn)
print("-----------------------------------")
print("SENSITIVITY   : ", se)
print("SPECIFICITY   : ", sp)
print("PRECISION     : ", ppv)
print("ACCURACY      : ", acc)
print("-----------------------------------")
