echo 'AV-Agent start'
start_time=$(date +%s)
start_time_str=$(date '+%Y-%m-%d %H:%M:%S')
echo "开始时间: $start_time_str"

python Main_run.py  --train /home/changxiaosong/python/malwareTest/little_train_youmi.txt --test /home/changxiaosong/python/malwareTest/little_test_youmi.txt
python Main_run.py  --train /home/changxiaosong/python/malwareTest/little_train_smssend.txt --test /home/changxiaosong/python/malwareTest/little_test_smssend.txt
python Main_run.py  --train /home/changxiaosong/python/malwareTest/little_train_smsreg.txt --test /home/changxiaosong/python/malwareTest/little_test_smsreg.txt
python Main_run.py  --train /home/changxiaosong/python/malwareTest/little_train_airpush.txt --test /home/changxiaosong/python/malwareTest/little_test_airpush.txt
python Main_run.py  --train /home/changxiaosong/python/malwareTest/little_train_kuguo.txt --test /home/changxiaosong/python/malwareTest/little_test_kuguo.txt
python Main_run.py  --train /home/changxiaosong/python/malwareTest/little_train_fakeinst.txt --test /home/changxiaosong/python/malwareTest/little_test_fakeinst.txt
python Main_run.py  --train /home/changxiaosong/python/malwareTest/little_train_dowgin.txt --test /home/changxiaosong/python/malwareTest/little_test_dowgin.txt

end_time=$(date +%s)
end_time_str=$(date '+%Y-%m-%d %H:%M:%S')
echo "结束时间: $end_time_str"

elapsed=$((end_time - start_time))
hours=$((elapsed / 3600))
minutes=$(( (elapsed % 3600) / 60 ))
echo "总耗时: ${hours}小时${minutes}分钟"