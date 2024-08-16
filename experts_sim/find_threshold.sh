


noise=0.1
threshold=0.9

for dim in {2..24}
# for dim in {21..21}
do
	maxclusters="4096"
	minclusters="2"
	clusters=$((($maxclusters + $minclusters) / 2))
	found="false"
	while [ "$found" != "true" ]
	do
		outfile="cluster_collisions/d${dim}_c${clusters}_n${noise}.txt"
		python cluster_collisions.new.py --sample_size 512 --data_dimension $dim --num_clusters $clusters --noise_level $noise --num_iters 3 --use_layer_norm >& $outfile
		cat $outfile
		misclass=`cat $outfile | cut -f 8 -d ' '`
		misclass=`echo "$misclass > $threshold" | bc`
		if [ "$misclass" == "0" ]
		then
			# Below the threshold, decrease clusters
			maxclusters="$clusters"
		else
			minclusters="$clusters"
		fi
		clusters=$((($maxclusters + $minclusters) / 2))
		if [ "$clusters" == "$maxclusters" ] && [ "$clusters" == "$minclusters" ]
		then
			found="true"
		fi
	done
	echo ""
done
