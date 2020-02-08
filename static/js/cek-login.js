var usernameSess;

function cekLogStatus(){
	$.ajax({
		type : 'POST',
		url : '/cek_user'
	}).done(function(data){
		console.log(data);
		if (data.auth == '0' || data.auth == '2' ) {
			console.log('not loged in');
			
			if (data.auth =='2') {
				$('#user-list').show();
				$('#log-element').empty();
				$('#log-element').append('<h3 style="text-align: right; margin-top: 50px;"  onclick="logOut()"><a href="#">Sign Out</a></h3>	');
			}else{
				$('#user-list').hide();
				$('#log-element').empty();
				$('#log-element').append('<h3 style="text-align: right; margin-top: 50px;"><a href="sign">Sign In</a></h3>	');
			}
			$('#sign-element').empty();
			$('#sign-element').append('Silahkan sign in untuk mengakses fitur lain</h4>');
			$('#estimate-box').hide();
			$('#no-access').text('Silahkan sign in sebagai user untuk mengakses fitur ini').show();

		} 
		else{
			usernameSess = data.user;
			$('#user-list').hide();
			$('#log-element').empty();
			$('#log-element').append('<h3 style="text-align: right; margin-top: 50px;" onclick="logOut()"><a href="#">Sign out</a></h3>	');
			$('#sign-element').empty();
			$('#sign-element').append('<h4>Hello, '+ data.user +'</h4>');
			$('#estimate-box').show();
			$('#latih-content').show();
			$('#no-access').hide();
			console.log('username = ', usernameSess);

		}
	});
}

function logOut(){
	
	$.ajax({
		type : 'POST',
		url : '/sign_out'
	}).done(function(data){
		console.log('logOut');
		cekLogStatus();
		location.reload();
	});

}