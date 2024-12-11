import os
import psutil
import sys
import time

def terminate_gpu_processes(process_ids, force=False):
    """
    Safely terminate specific GPU processes.
    
    Args:
        process_ids (list): List of process IDs to terminate
        force (bool): If True, uses SIGKILL instead of SIGTERM
    """
    successful = []
    failed = []
    
    for pid in process_ids:
        try:
            process = psutil.Process(int(pid))
            print(f"Attempting to terminate process {pid}")
            
            # First try graceful termination
            if force:
                process.kill()  # SIGKILL
            else:
                process.terminate()  # SIGTERM
            
            # Wait for the process to actually terminate
            try:
                process.wait(timeout=3)
                successful.append(pid)
                print(f"Successfully terminated process {pid}")
            except psutil.TimeoutExpired:
                if not force:
                    print(f"Process {pid} didn't terminate gracefully, trying force kill...")
                    process.kill()
                    process.wait(timeout=3)
                    successful.append(pid)
                    print(f"Successfully force killed process {pid}")
                else:
                    failed.append(pid)
                    print(f"Failed to terminate process {pid}")
                    
        except psutil.NoSuchProcess:
            print(f"Process {pid} does not exist")
            failed.append(pid)
        except psutil.AccessDenied:
            print(f"Access denied when trying to terminate process {pid}")
            failed.append(pid)
        except Exception as e:
            print(f"Error terminating process {pid}: {str(e)}")
            failed.append(pid)
    
    return successful, failed

if __name__ == "__main__":
    # The specific processes from your GPU
    gpu_processes = [261835]
    
    print("Starting process termination...")
    successful, failed = terminate_gpu_processes(gpu_processes)
    
    print("\nSummary:")
    print(f"Successfully terminated: {len(successful)} processes")
    if failed:
        print(f"Failed to terminate: {len(failed)} processes")
        print("Failed PIDs:", failed)
        sys.exit(1)