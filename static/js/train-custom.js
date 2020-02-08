var loss;
var mae;
var valLoss;
var valMae;
var color;
var inpaint;


function trainCustom(){
	var project = projectName;
	console.log(project)
	var ip = $('#ip-cek').is(":checked");
	var boxes = $('input[name=color-cek]:checked');

	
	console.log(boxes.length);
	var arrColor = []


	for (var i = 0; i < boxes.length; i++) {

		arrColor.push(boxes[i].value);
	}

	if (ip) {
		inpaint = 'yes';
	} else {
		inpaint = 'no';
	}

	console.log(arrColor);

	var warna = JSON.stringify(arrColor);
	console.log(color);

	$('#loading').show();
	$('#process-wait').text("Tunggu sebentar, sedang proses latih..").show();
	$('#train-result').hide();
	$('#save-sukses').hide();

	$.ajax({
		data: {
			x : arrColor,
			y : warna,
			ip : ip,
			project : project
		},
		type : 'POST',
		url : '/do_train_custom',
		
	}).done(function(data){
		
		$('#loading').hide();
		$('#process-wait').hide();
			if (data.success) {
				$('#train-result').show();
				$('#color-space').text(data.color).show();
				$('#inpaint-status').text(data.inpaint).show();
				$('#loss').text(data.loss).show();
				$('#mae').text(data.mae).show();
				$('#val-loss').text(data.valLoss).show();
				$('#val-mae').text(data.valMae).show();
				loss = data.loss;
				mae = data.mae;
				valLoss = data.valLoss;
				valMae = data.valMae;
				color = data.warna;
				
			} else {
				$('#process-wait').text("Mohon maaf sedang terjadi masalah").show();
			}

	});
		
}

function colorCheck(id){
	var boxes = $('input[name=color-cek]:checked');
	console.log(boxes.length);
	if (boxes.length == 3) {
		console.log('max');
		console.log(id.id);
		$('#'+id.id).prop('checked', false);
	}
}
