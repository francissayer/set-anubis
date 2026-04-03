import sys, os
import numpy as np
from glob import glob
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("--jobscriptDir", type=str, default="")
parser.add_argument("--memory", default = "6G")
parser.add_argument("--dryrun", action="store_true", help="Run without submitting, but produces the jobscripts and submission files which can be checked.")
parser.add_argument("--testArg", action="store_true", help="Just a proxy argument for reference")
args = parser.parse_args()


bashString = f"""#!/bin/bash

cd /usera/fs568/set-anubis
source /usera/fs568/set-anubis/.venv/bin/activate
"""

commandString = f"/usera/fs568/set-anubis/.venv/bin/python /usera/fs568/set-anubis/setanubis/FINAL_Z_Analysis_With_Reweighting/pp_production/uu_Decay_Channel/1_alp_Z_MadGraph_Interface+df_creation+df_to_sampledfs.py"

if args.testArg:
    commandString+=f" --test"

print("=============")
print(f"The base bash string is:\n {bashString} {commandString}")

if input("Confirm this is correct with y...").lower() != "y":
    raise Exception(f"Incorrect base string, please adjust your arguments")


if args.jobscriptDir=="":
    jobscriptDir = f"./jobscripts/"
else:
    jobscriptDir=args.jobscriptDir

if not os.path.exists(jobscriptDir):
    os.makedirs(jobscriptDir)

# Could put the following in a loop for a range of masses and couplings etc
for jobID in [15]:
    jobDir = f"{jobscriptDir}/job{jobID}"

    if not os.path.exists(jobDir):
        os.makedirs(jobDir)
    else:
        t = input(f"WARNING: A job with this ID ({jobID}) already exists, do you want to overwrite it? [y/n]")
        if t.lower() == "y":
            os.system(f"rm -r {jobDir}/*") # Remove things in jobDir if it exists to ensure unique runs.
        else:
            print(f"Skipping this job ID: {jobID}...")
            continue

    # Create Bash script
    bashScriptName = f"{jobDir}/runJob" 
    
    with open(f"{bashScriptName}.sh","w") as f:
        f.write(bashString)

        tempString = commandString + f" --secondTestCommand"

        f.write(tempString)

    # Create condor submission script
    condorString = f"executable = {bashScriptName}.sh" + "\n"
    condorString+= f"output = {jobDir}/job{jobID}_output.log" + "\n"
    condorString+= f"error =  {jobDir}/job{jobID}_error.log" + "\n"
    condorString+= f"request_memory = {args.memory}" + "\n"
    condorString+= f"log =  {jobDir}/job{jobID}_log.log" + "\n"
    condorString+= "copy_to_spool = true\n"
    condorString+= "should_transfer_files = YES\n"
    condorString+= "when_to_transfer_output = ON_EXIT_OR_EVICT\n"
    condorString+= "Queue"
    
    condorSubmissionFile = f"{jobDir}/condor_submit.job"
    with open(condorSubmissionFile,"w") as c:
        c.write(condorString)

    print(f"condor_submit {condorSubmissionFile}")
    if not args.dryrun:
        os.system(f"condor_submit {condorSubmissionFile}")
