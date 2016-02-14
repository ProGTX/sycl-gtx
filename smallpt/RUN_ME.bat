@echo off

echo Tester for smallpt in SYCL
del output.txt

echo Starting sycl-gtx program, no OpenCL
echo .\smallpt.exe 20 0 2 2>&1 | tee.exe -a output.txt

echo Starting sycl-gtx program, OpenCL only
.\smallpt.exe 20 2 2>&1 | tee.exe -a output.txt

echo Starting ComputeCpp program
.\smallpt-ccpp.exe 20 2>&1 | tee.exe -a output.txt
