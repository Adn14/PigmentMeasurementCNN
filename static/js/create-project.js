var projectName;
var projectId;

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
			$('#setting-custom').show();
			projectId = data.id;
			console.log(projectId);
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
		console.log(data.projects);
		var project = [];
		$('#list-project').empty();
		$('#list-project').append('<tr>	<th>Nama Project</th><th>Creator</th><th>Tombol Pilih</th>	</tr>	');
		for(i=0; i<data.projects.length; i++){
			var id = data.projects[i][0]
			var pro = data.projects[i][1]
			var creator = data.projects[i][2]
			console.log(id, pro, creator);
			$('#list-project').append('<tr><td  >'+pro+'</td><td>'+creator+'</td>'+
				'<td><button class="btn" onclick=enterProject("'+pro+'","'+id+'") >Pilih</button></td></tr>');
		}

	});
}

function enterProject(str, id){
	projectName = str;	
	projectId = id;
	$('#nama-project').text(projectName).show();
	$('#container-list').hide();
	//$('#setting').show();
	$('#setting-custom').show();
	console.log(id);
	
}

function saveTrainResult(){
	console.log(projectName);
	console.log(projectId);
	console.log(color);
	console.log(inpaint);
	$.ajax({
		data : {
			project : projectName,
			id : projectId,
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