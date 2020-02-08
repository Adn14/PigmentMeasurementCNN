function showSignUp(){
	$('#signin-box').hide();
	$('#signup-box').show();
}



function signIn(name, pass){
	if (name == '' || pass == '') {
		console.log('no empty');
		$('#errorAlert').text('Sign in gagal, input tidak boleh kosong').show();
	} else {
		$.ajax({
			data : {
				name : name,
				pass : pass,				
			},
			type : 'POST',
			url : '/sign_in'

		}).done(function(data){
			console.log(data.sukses);
			if(data.sukses == 'sukses'){
				console.log(data.sukses, data.nama, data.id);
				$('#successAlert').text('Sign in sukses. Redirecting..').show();
				$('#errorAlert').hide();
				$('#signin-box').hide();
				setTimeout(3000);
				window.location.replace("/");
			}else{
				console.log(data.sukses);
				$('#errorAlert').text('Sign up gagal, username atau password salah').show();
			}

		});

	}
	console.log(name, pass);
}

function signUp(name, pass){
	if (name == '' || pass == '') {
		console.log('no empty');
		$('#errorAlert').text('Sign up gagal, input tidak boleh kosong').show();
	} else {
		$.ajax({
			data : {
				name : name,
				pass : pass,				
			},
			type : 'POST',
			url : '/sign_up'

		}).done(function(data){
			console.log(data.sukses);
			if(data.sukses == 'sukses'){
				console.log(data.sukses);
				$('#successAlert').text('Sign up Sukses').show();
				$('#errorAlert').hide();
				$('#signin-box').show();
				$('#signup-box').hide();
			}else{
				console.log(data.sukses);
				$('#errorAlert').text('Sign up gagal, username sudah ada').show();
			}

		});

	}
	console.log(name, pass);
	
}