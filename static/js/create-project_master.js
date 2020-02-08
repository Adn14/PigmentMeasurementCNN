var projectName;

function noenter() {
  return !(window.event && window.event.keyCode == 13); }


function showCreateProject(){
	$('#project-option').hide();
	$('#create-project').show();

}

//creating new project in database of project
function createProject(){

	var name = $('#project-name').val();
	projectName = name;
	$.ajax({
		data : { 
			project : name
		},
		type : 'POST',
		url : '/create_project'
	})
	.done(function(data) {
		if (data.sukses) {
			$('#create-project').hide();
			$('#nama-project').text(name).show();
			$('#setting').show();
		}else{
			$('#name-exist').show();
		}
				
	});
}


function showProject(){
	$('#project-option').hide();
	$('#container-list').show();
	$.ajax({
		type : 'POST',
		url : '/show_project'
	})
	.done(function(data) {
		var project = [];
		$('#list-project').empty();
		$('#list-project').append('<tr>	<th>Nama Project</th><th>Tombol Pilih</th>	</tr>	');
		for(i=0; i<data.project.length; i++){
			project.push(data.project[i]);
			console.log(project[i]["0"]);
			var pro = String(project[i]["0"]);
			$('#list-project').append('<tr><td  >'+project[i]+'</td>'+
				'<td><button class="btn" onclick=enterProject("'+pro+'") >Pilih</button></td></tr>');
		}

	});
}

function enterProject(str){
	projectName = str;	
	$('#nama-project').text(projectName).show();
	$('#container-list').hide();
	$('#setting').show();
	console.log(str);
	
}

function saveTrainResult(){
	console.log(projectName);
	console.log(inpaint);
	console.log(loss);
	console.log(mae);
	console.log(valLoss);
	console.log(valMae);
	$.ajax({
		data : {
			project : projectName,
			color : color,
			mode : inpaint,
			loss : loss,
			mae : mae,
			valLoss : valLoss,
			valMae : valMae
		},
		type : 'POST',
		url : '/saveTrResult'
	})
	.done(function(data){
		console.log(data.project);
		$('#save-sukses').show();
	});

}