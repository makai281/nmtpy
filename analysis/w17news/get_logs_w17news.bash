#!/bin/bash

mdir=/lium/trad4a/wmt/2017/news/barrault/models/

cwd=`pwd`

logs=(
wmt17news-bpe-en-de.big/attention-e512-r1024-adam_4e-04-bs32-meteor-each5000-l2_1e-05-gc5-init_xavier-s1235-new5k.1.log
wmt17news-bpe-en-de.big/attention-e512-r1024-adam_4e-04-bs32-meteor-each1000-l2_1e-05-gc5-init_xavier-s1235.6.log
wmt17news-bpe-en-de/attention-e100-r100-adam_4e-04-bs32-meteor-each1000-l2_1e-05-do_0.2_0.4_0.4-gc5-init_xavier-s1235.15.log
wmt17news-bpe-en-de-notc.big/attention-e512-r1024-adam_4e-04-bs32-meteor-each5000-l2_1e-05-gc5-init_xavier-s1235-compare_tc_notc.1.log
wmt17news-bpe-en-de.big+btUEDIN/attention-e512-r1024-adam_4e-04-bs32-meteor-each5000-l2_1e-05-gc5-init_xavier-s1235.1.log
)

echo -n 'logfilenames = ( '

for l in ${logs[*]}
do
	rsync -av localhost:$mdir/$l ./ 2>&1 > /dev/null

	base=`basename $l`
	echo '"'`basename $cwd`/$base'",' 

done

echo ' )'


