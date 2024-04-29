import os
import logging

def get_logger(model_name, task_name, precision):
    logs_dir = os.path.join(os.getcwd(), 'logs', model_name) 
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    file_path = os.path.join(logs_dir, f"{task_name}_{precision}_logs.log") if precision else os.path.join(logs_dir, f"{task_name}_logs.log")

    logging.basicConfig(filename=file_path, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)

    return logger

def write_profiling_results(prof, file_path, processor_type="cpu", profiling_type="time", row_limit=10, is_quantized=False):
    profile_type = f"{processor_type}_time_total" if profiling_type == "time" else f"{processor_type}_memory_usage"
    quantization_status = "quantized" if is_quantized else "unquantized"

    file_path = os.path.join(file_path, quantization_status, f"sorted_{profiling_type}")
    if not os.path.exists(file_path):
        os.makedirs(file_path)
    file_path = os.path.join(file_path, f"{processor_type}_profiling_results.txt")
    
    with open(file_path, "w") as file:
        file.write(prof.key_averages().table(sort_by=profile_type, row_limit=row_limit))

def log_profiling_metrics(prof, model, task, logger, precision, file_path="profiling_results", is_quantized=False):
    currpath = os.path.abspath(os.curdir)
    file_path = os.path.join(currpath, 'results', file_path, model, precision, task) if precision else os.path.join(currpath, 'results', file_path, model, task)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    logger.info("Profiling results...")
    write_profiling_results(prof, file_path, processor_type="cpu", profiling_type="time", is_quantized=is_quantized)
    write_profiling_results(prof, file_path, processor_type="cuda", profiling_type="time", is_quantized=is_quantized)
    write_profiling_results(prof, file_path, processor_type="cpu", profiling_type="memory", is_quantized=is_quantized)
    write_profiling_results(prof, file_path, processor_type="cuda", profiling_type="memory", is_quantized=is_quantized)
    
    logger.info(f"Profiling results for {model} and {task} are saved in {file_path} folder.")

def write_performance_results_to_file(model_name, model_size, accuracy, quantized_size, quantized_accuracy, task_name, logger, precision):
    currpath = os.path.abspath(os.curdir)
    results_dir = os.path.join(currpath, 'results', 'performance_results', model_name, precision) if precision else os.path.join(currpath, 'results', 'performance_results', model_name)

    os.makedirs(results_dir, exist_ok=True)
    file_path = os.path.join(results_dir, f'{task_name}.txt')
    
    with open(file_path, 'w') as f:
        f.write(f"Size before Quantization (GB) : {model_size}\n")
        f.write(f"Accuracy before quantization : {accuracy}\n")
        f.write(f"-----------------------------------\n")
        f.write(f"Size after Quantization (GB): {quantized_size}\n")
        f.write(f"Accuracy after quantization : {quantized_accuracy}")
        f.close()

        logger.info(f"Quantization results are saved at {currpath}/results/performance_results/{model_name}/{task_name}.txt")