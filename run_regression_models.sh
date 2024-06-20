nohup Rscript --vanilla analyses/baseline_analyses.R -r FPRT -i 6000 > logs/FPRT.log&
nohup Rscript --vanilla analyses/baseline_analyses.R -r TFT -i 6000 > logs/TFT.log&
nohup Rscript --vanilla analyses/baseline_analyses.R -r Fix -i 6000 > logs/Fix.log&
nohup Rscript --vanilla analyses/baseline_analyses.R -r FPReg -i 6000 > logs/FPReg.log&