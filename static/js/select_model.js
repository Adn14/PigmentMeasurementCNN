

//js for estimate.html untuk model selection

$(this).scrollTop(0);
function showSelectedModel(){
	$(this).scrollTop(0);
	$.ajax({
		type : 'POST',
		url : '/get_selected_model',
	}).done(function(data){
		$('#btn-color-model').hide();
		$('#btn-available').hide();
		$('#selected-model').show();
		console.log(data.selected[0]);
		$('#selected-model').empty();
		$('#selected-model').append('<tr>	<th colspan="7"> Pilihan Model </th>	</tr>	');
		$('#selected-model').append('<tr>	<th>No</th><th>Warna</th><th>Inpaint</th><th>Project</th><th>File</th><th>Timestamp</th> <th></th	</tr>	');
		for(i=0; i<data.selModel.length; i++){	
			console.log('table made');			
			let warna = String(data.selModel[i][0]);
			let inpaint = String(data.selModel[i][1]);
			let project = String(data.selModel[i][2]);
			let file = String(data.selModel[i][3]);
			let time = String(data.selModel[i][4]);
			$('#selected-model').append('<tr> <td  >'+ (i+1) +'</td><td  >'+warna+'</td> <td >'+inpaint+'</td> <td  >'+project+'</td><td  >'+file+'</td> <td  >'+time+'</td> <td  > <button class="btn" onclick=showAvailableProject("'+warna+'","'+inpaint+'")>Cari</button></td> </tr>');
		}
		
		console.log(data.selModel);
		console.log('finish');
	});
}

function backSelModel(){
	$(this).scrollTop(0);
	showSelectedModel();
	$('#btn-color-model').hide();
	$('#color-model').hide();
}

function backAvailModel(){
	$(this).scrollTop(0);
	$('#color-model').show();
	$('#available-model').hide();
	$('#btn-color-model').show();
	$('#btn-available').hide();
}

function showAvailableProject(warna, status){
	$(this).scrollTop(0);
	console.log('show select')
	$('#btn-color-model').show();
	$('#btn-available').hide();
	$('#btn-to-estimate').hide();
	color = warna;
	inpaint = status;
	var projects = [];

	$.ajax({
		data : {color : color, inpaint : inpaint},
		type : 'POST',
		url : '/get_available_project',
		async:false
	}).done(function(data){
		projects = data;
		console.log(data.available);
		if (data.available.length != 0) {
			$('#selected-model').hide();
			$('#color-model').show();
			$('#color-model').empty();
			$('#color-model').append('<tr>	<th colspan="4"> Model yang tersedia </th>	</tr>	');
			$('#color-model').append('<tr>	<th>No</th><th>Project</th><th>Project creator</th><th></th	</tr>	');
			for(i=0; i<data.available.length; i++){	
				console.log('table made');			
				let project = String(data.available[i]);
				let creator = String(data.creator[i]);
				
				$('#color-model').append('<tr> <td  >'+ (i+1) +'</td><td  >'+project+'</td> <td  >'+creator+'</td><td  > <button class="btn" onclick=pilihModel("'+project+'","'+color+'","'+inpaint+'")>Pilih</button></td> </tr>');
			}
		} else{
			$('#color-model').empty();
			$('#color-model').append('<tr>	<th colspan="3"> Model yang tersedia </th>	</tr>	');
			$('#color-model').append('<tr>	<th>No</th><th>Project</th><th></th	</tr>	');
			$('#color-model').append('<tr>	<td colspan="3"> Belum dilakukan training </td>	</tr>	');
		}
	});
}

function pilihModel(project, warna, status){
	$(this).scrollTop(0);
	console.log(project + warna + status);
	$('#btn-color-model').hide();
	$('#btn-available').show();
	var proj = project;
	var color = warna;
	var inpaint = status;

	$.ajax({
		data : {project : proj, color : color, inpaint : inpaint},
		type : 'POST',
		url : '/pilih_model',
	}).done(function(data){
		console.log(data.models);
		$('#color-model').hide();
		$('#available-model').show();
		$('#available-model').empty();
		$('#available-model').append('<tr>	<th colspan="5"> '+ data.project + ' - ' + data.color +' </th>	</tr>	');
		$('#available-model').append('<tr>	<th>No</th><th>Model</th><th>Creator</th><th></th	</tr>	');
		for(i = 0; i < data.models.length; i++){
			let project = data.project;
			let thisColor = data.color;
			let model = data.models[i]
			let creator = data.creator[i]
			$('#available-model').append('<tr> <td  >'+ (i+1) +'</td><td  >'+model+'</td><td  >'+creator+'</td><td  > <button class="btn" onclick=fixPilihModel("'+project+'","'+color+'","'+model+'","'+inpaint+'")>Pilih</button></td> </tr>');

		}

	});
}

function fixPilihModel(project, warna, model, inpaint){
	$(this).scrollTop(0);
	console.log(project, warna, inpaint, model);

	$.ajax({
		data : {project : project, color : warna, inpaint : inpaint, model},
		type : 'POST',
		url : '/fix_pilih_model',
	}).done(function(data){
		$('#available-model').hide();
		$('#btn-to-estimate').show();
		showSelectedModel();
	});
}