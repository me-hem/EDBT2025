min_slot=0
max_slot=8000000
read -p "Enter the desired block number: " desired_block

while [[ $min_slot -le $max_slot ]]; do
	mid=$(( ($min_slot + $max_slot) / 2 ))
	block_num=$(curl -s http://localhost:3500/eth/v2/beacon/blocks/$mid | jq -r '.data.message.body.execution_payload.block_number' | xargs -I {} geth attach --exec "eth.getBlock({}).number" http://localhost:8545)
	
	if [[ $desired_block -eq $block_num ]]; then
		echo "Slot $mid => Block No. $desired_block found successfully!"
	        exit 0
	elif [[ $desired_block -gt $block_num ]]; then
		min_slot=$(( $mid + 1 ))
	else
		max_slot=$(( $mid - 1 ))		
	fi
	
	echo "Slot $slot => Block No. $block_num processed."
done

echo "Mapping for Block No. $desired_block not found!"	
