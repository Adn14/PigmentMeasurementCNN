//for record html
$(this).scrollTop(0);

let currentProject;

function backRecord(){
	$('#project-list').show();
	$('#title-record').text('List Project').show();
	$('#scores').hide();
	$('#back-btn').hide();
}

function showProjectList(){
	console.log('load list');

	
	$.ajax({
		type : 'POST',
		url : '/show_project'
	})
	.done(function(data) {
		var project = [];
		
		$('#project-list').empty();
		$('#project-list').append('<tr>	<th>Nama Project</th><th>Creator</th><th>Tombol Pilih</th>	</tr>	');
		for(i=0; i<data.projects.length; i++){
			project.push(data.projects[i]);
			console.log(project[i][1]);
			var pro = data.projects[i][1]
			var creator = data.projects[i][2]
			$('#project-list').append('<tr><td  ><h5 style="font-style: bold;">'+pro+'</h5></td>'+ '<td>'+ creator +'</td>' +
				'<td><button class="btn" onclick=enterProjectScores("'+pro+'") >Pilih</button></td></tr>');
		}

	});
}

function getCsv(){
	let project = currentProject;

	$.get('static/project_record/'+project+'skor.csv')
	.done(function(csvData) {
		var uri = 'data:application/csv;charset=UTF-8,' + encodeURIComponent(csvData);
		var downloadLink = document.createElement("a");
		downloadLink.href = uri;
		downloadLink.download = project + ".csv";

		document.body.appendChild(downloadLink);
		downloadLink.click();
		document.body.removeChild(downloadLink);
		});	
}

function enterProjectScores(project){
	currentProject = project;
	$('#loading').show();
	$('#project-list').hide();
	console.log('load project');
	$.ajax({
		data : {project : project},
		type : 'POST',
		url : '/show_scores'

	})
	.done(function(data){
		$('#loading').hide();
		$('#title-record').text('Project ' + data.project + ' scores data').show();
		$('#back-btn').show();
		$('#scores').show();
		populateScoreTable(data.rgb, 'rgb', false);
		populateScoreTable(data.rgbHsv, 'rgbhsv', false);
		populateScoreTable(data.rgbLab, 'rgblab', false);
		populateScoreTable(data.rgbYcbcr, 'rgbycb', false);
		populateScoreTable(data.hsv, 'hsv', false);
		populateScoreTable(data.hsvLab, 'hsvlab', false);
		populateScoreTable(data.hsvYcbcr, 'hsvycb', false);
		populateScoreTable(data.lab, 'lab', false);
		populateScoreTable(data.labYcbcr, 'labycb', false);
		populateScoreTable(data.ycbcr, 'ycb', false);

		populateScoreTable(data.rgbIp, 'iprgb', true);
		populateScoreTable(data.rgbHsvIp, 'iprgbhsv', true);
		populateScoreTable(data.rgbLabIp, 'iprgblab', true);
		populateScoreTable(data.rgbYcbcrIp, 'iprgbycb', true);
		populateScoreTable(data.hsvIp, 'iphsv', true);
		populateScoreTable(data.hsvLabIp, 'iphsvlab', true);
		populateScoreTable(data.hsvYcbcrIp, 'iphsvycb', true);
		populateScoreTable(data.labIp, 'iplab', true);
		populateScoreTable(data.labYcbcrIp, 'iplabycb', true);
		populateScoreTable(data.ycbcrIp, 'ipycb', true);

		console.log(data.avRgb[0])
		$('#average-scores').empty();
		$('#average-scores').append('<tr>	<th colspan="6"> Rata-rata skor </th>	</tr>	');
		$('#average-scores').append('<tr>	<th>Ruang Warna</th> <th>Inpainted</th> <th>Training Loss</th> <th>Training MAE</th><th>Validation Loss</th><th>Validation MAE</th>	</tr>	');
		$('#average-scores').append('<tr>	<td>RGB</td> <td>No</td> <td>'+  ((data.avRgb[0] == '0' ) ? '-' : data.avRgb[0] ) +'</td> <td>'+ ((data.avRgb[1] == '0' ) ? '-' : data.avRgb[1] ) +'</td><td>'+ ((data.avRgb[2] == '0' ) ? '-' : data.avRgb[2] ) +'</td><td>'+ ((data.avRgb[3] == '0' ) ? '-' : data.avRgb[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>RGB + HSV</td> <td>No</td> <td>'+  ((data.avRgbHsv[0] == '0' ) ? '-' : data.avRgbHsv[0] ) +'</td> <td>'+ ((data.avRgbHsv[1] == '0' ) ? '-' : data.avRgbHsv[1] ) +'</td><td>'+ ((data.avRgbHsv[2] == '0' ) ? '-' : data.avRgbHsv[2] ) +'</td><td>'+ ((data.avRgbHsv[3] == '0' ) ? '-' : data.avRgbHsv[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>RGB + LAB</td> <td>No</td> <td>'+  ((data.avRgbLab[0] == '0' ) ? '-' : data.avRgbLab[0] ) +'</td> <td>'+ ((data.avRgbLab[1] == '0' ) ? '-' : data.avRgbLab[1] ) +'</td><td>'+ ((data.avRgbLab[2] == '0' ) ? '-' : data.avRgbLab[2] ) +'</td><td>'+ ((data.avRgbLab[3] == '0' ) ? '-' : data.avRgbLab[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>RGB + YCbCr</td> <td>No</td> <td>'+  ((data.avRgbYcb[0] == '0' ) ? '-' : data.avRgbYcb[0] ) +'</td> <td>'+ ((data.avRgbYcb[1] == '0' ) ? '-' : data.avRgbYcb[1] ) +'</td><td>'+ ((data.avRgbYcb[2] == '0' ) ? '-' : data.avRgbYcb[2] ) +'</td><td>'+ ((data.avRgbYcb[3] == '0' ) ? '-' : data.avRgbYcb[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>HSV</td> <td>No</td> <td>'+  ((data.avHsv[0] == '0' ) ? '-' : data.avHsv[0] ) +'</td> <td>'+ ((data.avHsv[1] == '0' ) ? '-' : data.avHsv[1] ) +'</td><td>'+ ((data.avHsv[2] == '0' ) ? '-' : data.avHsv[2] ) +'</td><td>'+ ((data.avHsv[3] == '0' ) ? '-' : data.avHsv[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>HSV + LAB</td> <td>No</td> <td>'+  ((data.avHsvLab[0] == '0' ) ? '-' : data.avHsvLab[0] ) +'</td> <td>'+ ((data.avHsvLab[1] == '0' ) ? '-' : data.avHsvLab[1] ) +'</td><td>'+ ((data.avHsvLab[2] == '0' ) ? '-' : data.avHsvLab[2] ) +'</td><td>'+ ((data.avHsvLab[3] == '0' ) ? '-' : data.avHsvLab[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>HSV + YCbCr</td> <td>No</td> <td>'+  ((data.avHsvYcb[0] == '0' ) ? '-' : data.avHsvYcb[0] ) +'</td> <td>'+ ((data.avHsvYcb[1] == '0' ) ? '-' : data.avHsvYcb[1] ) +'</td><td>'+ ((data.avHsvYcb[2] == '0' ) ? '-' : data.avHsvYcb[2] ) +'</td><td>'+ ((data.avHsvYcb[3] == '0' ) ? '-' : data.avHsvYcb[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>LAB</td> <td>No</td> <td>'+  ((data.avLab[0] == '0' ) ? '-' : data.avLab[0] ) +'</td> <td>'+ ((data.avLab[1] == '0' ) ? '-' : data.avLab[1] ) +'</td><td>'+ ((data.avLab[2] == '0' ) ? '-' : data.avLab[2] ) +'</td><td>'+ ((data.avLab[3] == '0' ) ? '-' : data.avLab[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>LAB + YCbCr</td> <td>No</td> <td>'+  ((data.avLabYcb[0] == '0' ) ? '-' : data.avLabYcb[0] ) +'</td> <td>'+ ((data.avLabYcb[1] == '0' ) ? '-' : data.avLabYcb[1] ) +'</td><td>'+ ((data.avLabYcb[2] == '0' ) ? '-' : data.avLabYcb[2] ) +'</td><td>'+ ((data.avLabYcb[3] == '0' ) ? '-' : data.avLabYcb[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>YCBCr</td> <td>No</td> <td>'+  ((data.avYcbcr[0] == '0' ) ? '-' : data.avYcbcr[0] ) +'</td> <td>'+ ((data.avYcbcr[1] == '0' ) ? '-' : data.avYcbcr[1] ) +'</td><td>'+ ((data.avYcbcr[2] == '0' ) ? '-' : data.avYcbcr[2] ) +'</td><td>'+ ((data.avYcbcr[3] == '0' ) ? '-' : data.avYcbcr[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>RGB</td> <td>Yes</td> <td>'+  ((data.avRgbIp[0] == '0' ) ? '-' : data.avRgbIp[0] ) +'</td> <td>'+ ((data.avRgbIp[1] == '0' ) ? '-' : data.avRgbIp[1] ) +'</td><td>'+ ((data.avRgbIp[2] == '0' ) ? '-' : data.avRgbIp[2] ) +'</td><td>'+ ((data.avRgbIp[3] == '0' ) ? '-' : data.avRgbIp[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>RGB + HSV</td> <td>Yes</td> <td>'+  ((data.avRgbHsvIp[0] == '0' ) ? '-' : data.avRgbHsvIp[0] ) +'</td> <td>'+ ((data.avRgbHsvIp[1] == '0' ) ? '-' : data.avRgbHsvIp[1] ) +'</td><td>'+ ((data.avRgbHsvIp[2] == '0' ) ? '-' : data.avRgbHsvIp[2] ) +'</td><td>'+ ((data.avRgbHsvIp[3] == '0' ) ? '-' : data.avRgbHsvIp[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>RGB + LAB</td> <td>Yes</td> <td>'+  ((data.avRgbLabIp[0] == '0' ) ? '-' : data.avRgbLabIp[0] ) +'</td> <td>'+ ((data.avRgbLabIp[1] == '0' ) ? '-' : data.avRgbLabIp[1] ) +'</td><td>'+ ((data.avRgbLabIp[2] == '0' ) ? '-' : data.avRgbLabIp[2] ) +'</td><td>'+ ((data.avRgbLabIp[3] == '0' ) ? '-' : data.avRgbLabIp[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>RGB + YCbCr</td> <td>Yes</td> <td>'+  ((data.avRgbYcbIp[0] == '0' ) ? '-' : data.avRgbYcbIp[0] ) +'</td> <td>'+ ((data.avRgbYcbIp[1] == '0' ) ? '-' : data.avRgbYcbIp[1] ) +'</td><td>'+ ((data.avRgbYcbIp[2] == '0' ) ? '-' : data.avRgbYcbIp[2] ) +'</td><td>'+ ((data.avRgbYcbIp[3] == '0' ) ? '-' : data.avRgbYcbIp[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>HSV</td> <td>Yes</td> <td>'+  ((data.avHsvIp[0] == '0' ) ? '-' : data.avHsvIp[0] ) +'</td> <td>'+ ((data.avHsvIp[1] == '0' ) ? '-' : data.avHsvIp[1] ) +'</td><td>'+ ((data.avHsvIp[2] == '0' ) ? '-' : data.avHsvIp[2] ) +'</td><td>'+ ((data.avHsvIp[3] == '0' ) ? '-' : data.avHsvIp[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>HSV + LAB</td> <td>Yes</td> <td>'+  ((data.avHsvLabIp[0] == '0' ) ? '-' : data.avHsvLabIp[0] ) +'</td> <td>'+ ((data.avHsvLabIp[1] == '0' ) ? '-' : data.avHsvLabIp[1] ) +'</td><td>'+ ((data.avHsvLabIp[2] == '0' ) ? '-' : data.avHsvLabIp[2] ) +'</td><td>'+ ((data.avHsvLabIp[3] == '0' ) ? '-' : data.avHsvLabIp[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>HSV + YCbCr</td> <td>Yes</td> <td>'+  ((data.avHsvYcbIp[0] == '0' ) ? '-' : data.avHsvYcbIp[0] ) +'</td> <td>'+ ((data.avHsvYcbIp[1] == '0' ) ? '-' : data.avHsvYcbIp[1] ) +'</td><td>'+ ((data.avHsvYcbIp[2] == '0' ) ? '-' : data.avHsvYcbIp[2] ) +'</td><td>'+ ((data.avHsvYcbIp[3] == '0' ) ? '-' : data.avHsvYcbIp[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>LAB</td> <td>Yes</td> <td>'+  ((data.avLabIp[0] == '0' ) ? '-' : data.avLabIp[0] ) +'</td> <td>'+ ((data.avLabIp[1] == '0' ) ? '-' : data.avLabIp[1] ) +'</td><td>'+ ((data.avLabIp[2] == '0' ) ? '-' : data.avLabIp[2] ) +'</td><td>'+ ((data.avLabIp[3] == '0' ) ? '-' : data.avLabIp[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>LAB + YCbCr</td> <td>Yes</td> <td>'+  ((data.avLabYcbIp[0] == '0' ) ? '-' : data.avLabYcbIp[0] ) +'</td> <td>'+ ((data.avLabYcbIp[1] == '0' ) ? '-' : data.avLabYcbIp[1] ) +'</td><td>'+ ((data.avLabYcbIp[2] == '0' ) ? '-' : data.avLabYcbIp[2] ) +'</td><td>'+ ((data.avLabYcbIp[3] == '0' ) ? '-' : data.avLabYcbIp[3] ) +'</td>	</tr>	');
		$('#average-scores').append('<tr>	<td>YCBCr</td> <td>Yes</td> <td>'+  ((data.avYcbcrIp[0] == '0' ) ? '-' : data.avYcbcrIp[0] ) +'</td> <td>'+ ((data.avYcbcrIp[1] == '0' ) ? '-' : data.avYcbcrIp[1] ) +'</td><td>'+ ((data.avYcbcrIp[2] == '0' ) ? '-' : data.avYcbcrIp[2] ) +'</td><td>'+ ((data.avYcbcrIp[3] == '0' ) ? '-' : data.avYcbcrIp[3] ) +'</td>	</tr>	');
		
		$('#im-loss').attr("src","static/project_record/"+ data.imLoss);
		$('#im-lossIp').attr("src","static/project_record/"+ data.imLossIp);
		$('#im-mae').attr("src","static/project_record/"+ data.imMae);
		$('#im-maeIp').attr("src","static/project_record/"+ data.imMaeIp);


	});
}

function populateScoreTable(data, color, inpaint){
	$('#project-list').hide();
	if(color == 'rgb'){
		id = '#rgb-scores';
	} else if (color == 'rgbhsv'){
		id = '#rgbhsv-scores';
	} else if (color == 'rgblab'){
		id = '#rgblab-scores';
	}  else if (color == 'rgbycb'){
		color = 'rgbycbcr';
		id = '#rgbycb-scores';
	}  else if (color == 'hsv'){
		id = '#hsv-scores';
	}  else if (color == 'hsvlab'){
		id = '#hsvlab-scores';
	}  else if (color == 'hsvycb'){
		color = 'hsvycbcr';
		id = '#hsvycb-scores';
	}  else if (color == 'lab'){
		id = '#lab-scores';
	}  else if (color == 'labycb'){
		color = 'labycbcr';
		id = '#labycb-scores';
	}  else if (color == 'ycb'){
		color = 'ycbcr';
		id = '#ycb-scores';
	}  

	else if(color == 'iprgb'){
		id = '#iprgb-scores';
	} else if (color == 'iprgbhsv'){
		id = '#iprgbhsv-scores';
	} else if (color == 'iprgblab'){
		id = '#iprgblab-scores';
	}  else if (color == 'iprgbycb'){
		color = 'iprgbycbcr';
		id = '#iprgbycb-scores';
	}  else if (color == 'iphsv'){
		id = '#iphsv-scores';
	}  else if (color == 'iphsvlab'){
		id = '#iphsvlab-scores';
	}  else if (color == 'iphsvycb'){
		color = 'iphsvycbcr';
		id = '#iphsvycb-scores';
	}  else if (color == 'iplab'){
		id = '#iplab-scores';
	}  else if (color == 'iplabycb'){
		color = 'iplabycbcr';
		id = '#iplabycb-scores';
	}  else if (color == 'ipycb'){
		color = 'ipycbcr';
		id = '#ipycb-scores';
	}  

	let mode;
	let status;
	mode = 'inpaint';
	if(inpaint == true){
		 status = 'yes';
		 mode = 'Inpaint';
	} else{
		status = 'no';
		 mode = 'Non Inpaint';
	}

	console.log(data);

	if(data == 0){
		console.log('kosong');
		$(id).empty();
		$(id).append('<tr>	<th colspan="5">'+mode+' </th>	</tr>	');
		$(id).append('<tr>	<th>Training Loss</th><th>Training MAE</th><th>Validation Loss</th><th>Validation MAE</th><th>Timestamp</th>	</tr>	');
		$(id).append('<tr> <td colspan="5">Data Kosong</td> </tr>')
	} else{
		console.log(data[0][0]);
		$(id).empty();
		$(id).append('<tr>	<th colspan="8"> '+mode+' </th>	</tr>	');
		$(id).append('<tr>	<th>No</th><th>Training Loss</th><th>Training MAE</th><th>Validation Loss</th><th>Validation MAE</th><th>Timestamp</th><th>Creator</th> <th>Delete</th	</tr>	');
		for(i=0; i<data.length; i++){				
			let loss = String(data[i][0]);
			let mae = String(data[i][1]);
			let valLoss = String(data[i][2]);
			let valMae = String(data[i][3]);
			let time = String(data[i][4]);
			let creator = String(data[i][5]);
			$(id).append('<tr> <td  >'+ (i+1) +'</td><td  >'+loss+'</td> <td >'+mae+'</td> <td  >'+valLoss+'</td><td  >'+valMae+'</td> <td  >'+time+'</td><td  >'+creator+'</td> <td  > <button class="btn" onclick=delModel("'+color+'","'+time+'","'+status+'")>Hapus</button></td> </tr>');
		}

	}	
}

function delModel(colour, model, mode){
	let project = currentProject;
	let color = colour;
	let tStamp = model;

	
	console.log(color);
	//alert('refresh');
	//

	$.ajax({
		data : {
			project : project,
			mode : mode,
			color : color,
			tStamp : tStamp
		},
		type : 'POST',
		url : '/delete_model'
	})
	.done(function(data){
		enterProjectScores(currentProject);

	});

}