import os
import logging

def get_logger(log_file_path):
    logging.basicConfig(filename=log_file_path, filemode='w', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    return logger

def write_profiling_results(prof, model, task, processor_type="cpu", profiling_type="time", file_path="profiling_results", row_limit=10, is_quantized=False):
    profile_type = f"{processor_type}_time_total" if profiling_type == "time" else f"{processor_type}_memory_usage"
    quantization_status = "quantized" if is_quantized else "unquantized"
    file_path = os.path.join(file_path, f"{model}_{task}_sorted_{processor_type}_{profiling_type}_{quantization_status}_results.txt")
    
    with open(file_path, "w") as file:
        file.write(prof.key_averages().table(sort_by=profile_type, row_limit=row_limit))

def log_profiling_metrics(prof, model, task, logger, file_path="profiling_results", is_quantized=False):
    currpath = os.path.abspath(os.curdir)
    file_path = os.path.join(f'{currpath}/results', file_path)
    if not os.path.exists(file_path):
        os.makedirs(file_path)
        
    logger.info("Profiling results...")
    write_profiling_results(prof, model, task, processor_type="cpu", profiling_type="time", file_path=file_path, is_quantized=is_quantized)
    write_profiling_results(prof, model, task, processor_type="cuda", profiling_type="time", file_path=file_path, is_quantized=is_quantized)
    
    logger.info(f"Profiling results for {model} and {task} are saved in {file_path} folder.")

def write_quant_results_to_file(model_size, accuracy, quantized_size, quantized_accuracy, task_name, logger):
    currpath = os.path.abspath(os.curdir)
    with open(os.path.join(f'{currpath}/results', f'bert-base-{task_name}.txt'), 'w') as f:
        f.write(f"Size before Quantization (GB) : {model_size}\n")
        f.write(f"Accuracy before quantization : {accuracy}\n")
        f.write(f"-----------------------------------\n")
        f.write(f"Size after Quantization (GB): {quantized_size}\n")
        f.write(f"Accuracy after quantization : {quantized_accuracy}")
        f.close()

        logger.info(f"Quantization results are saved at {currpath}/results/bert-base-{task_name}.txt")