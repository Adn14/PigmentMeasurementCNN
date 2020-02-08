

function getUserList(){

	$.ajax({
		type : 'POST',
		url : '/get_user'

	}).done(function(data){
		console.log(data.user);
		$('#user-list').empty();
		$('#user-list').append('<tr>	<th colspan="3"> Daftar User </th>	</tr>	');
		$('#user-list').append('<tr>	<th>Id User</th><th>Username</th><th> </th>	</tr>	');
		for (var i = 0; i < data.user.length; i++) {
			let id = String(data.user[i][0]);
			let name = String(data.user[i][1]);
			
			$('#user-list').append('<tr> <td  >'+ id +'</td><td  >'+name+'</td><td  > <button class="btn" onclick=deleteUser("'+id+'")>Hapus User</button></td> </tr>');
			}
		});
}

function deleteUser(id){
	$.ajax({
		data : {
			id : id
		},
		type : 'POST',
		url : '/delete_user'

	}).done(function(data){
		console.log(data.sukses);
		getUserList();
	});
}