//js for estimasi.html

$(document).ready(function() {

	$('form').on('submit', function(event) {

		var formData = new FormData(this)
		$('#result').hide();
		$('#loading').show();
		$('#process-wait').show();
		$.ajax({
			data : formData,
			type : 'POST',
			url : '/estimate',
			cache: false,
            contentType: false,
            processData: false
		})
		.done(function(data) {
			console.log(data.img);

			if (data.success) {
				$('#successAlert').text(data.success).show();
				$('#result').show();

				$('#antosianin-rgb').text(data.npRgbAnto + ' abs/g').show();
				$('#karotenoid-rgb').text(data.npRgbKaro + ' μg/g').show();
				$('#klorofil-rgb').text(data.npRgbKlor + ' μg/g').show();

				$('#antosianin-rgbhsv').text(data.npRgbHsvAnto + ' abs/g').show();
				$('#karotenoid-rgbhsv').text(data.npRgbHsvKaro + ' μg/g').show();
				$('#klorofil-rgbhsv').text(data.npRgbHsvKlor + ' μg/g').show();

				$('#antosianin-rgblab').text(data.npRgbLabAnto + ' abs/g').show();
				$('#karotenoid-rgblab').text(data.npRgbLabKaro + ' μg/g').show();
				$('#klorofil-rgblab').text(data.npRgbLabKlor + ' μg/g').show();

				$('#antosianin-rgbycbcr').text(data.npRgbYcbcrAnto + ' abs/g').show();
				$('#karotenoid-rgbycbcr').text(data.npRgbYcbcrKaro + ' μg/g').show();
				$('#klorofil-rgbycbcr').text(data.npRgbYcbcrKlor + ' μg/g').show();

				$('#antosianin-hsv').text(data.npHsvAnto + ' abs/g').show();
				$('#karotenoid-hsv').text(data.npHsvKaro + ' μg/g').show();
				$('#klorofil-hsv').text(data.npHsvKlor + ' μg/g').show();

				$('#antosianin-hsvlab').text(data.npHsvLabAnto + ' abs/g').show();
				$('#karotenoid-hsvlab').text(data.npHsvLabKaro + ' μg/g').show();
				$('#klorofil-hsvlab').text(data.npHsvLabKlor + ' μg/g').show();

				$('#antosianin-hsvycbcr').text(data.npHsvYcbcrAnto + ' abs/g').show();
				$('#karotenoid-hsvycbcr').text(data.npHsvYcbcrKaro + ' μg/g').show();
				$('#klorofil-hsvycbcr').text(data.npHsvYcbcrKlor + ' μg/g').show();

				$('#antosianin-lab').text(data.npLabAnto + ' abs/g').show();
				$('#karotenoid-lab').text(data.npLabKaro + ' μg/g').show();
				$('#klorofil-lab').text(data.npLabKlor + ' μg/g').show();

				$('#antosianin-labycbcr').text(data.npLabYcbcrAnto+ ' abs/g').show();
				$('#karotenoid-labycbcr').text(data.npLabYcbcrKaro + ' μg/g').show();
				$('#klorofil-labycbcr').text(data.npLabYcbcrKlor + ' μg/g').show();
								
				$('#antosianin-ycbcr').text(data.npYCbCrAnto + ' abs/g').show();
				$('#karotenoid-ycbcr').text(data.npYCbCrKaro + ' μg/g').show();
				$('#klorofil-ycbcr').text(data.npYCbCrKlor + ' μg/g').show();

				$('#antosianin-ip-rgb').text(data.ipRgbAnto + ' abs/g').show();
				$('#karotenoid-ip-rgb').text(data.ipRgbKaro + ' μg/g').show();
				$('#klorofil-ip-rgb').text(data.ipRgbKlor + ' μg/g').show();

				$('#antosianin-ip-rgbhsv').text(data.ipRgbHsvAnto + ' abs/g').show();
				$('#karotenoid-ip-rgbhsv').text(data.ipRgbHsvKaro + ' μg/g').show();
				$('#klorofil-ip-rgbhsv').text(data.ipRgbHsvKlor + ' μg/g').show();

				$('#antosianin-ip-rgblab').text(data.ipRgbLabAnto + ' abs/g').show();
				$('#karotenoid-ip-rgblab').text(data.ipRgbLabKaro + ' μg/g').show();
				$('#klorofil-ip-rgblab').text(data.ipRgbLabKlor + ' μg/g').show();

				$('#antosianin-ip-rgbycbcr').text(data.ipRgbYcbcrAnto + ' abs/g').show();
				$('#karotenoid-ip-rgbycbcr').text(data.ipRgbYcbcrKaro + ' μg/g').show();
				$('#klorofil-ip-rgbycbcr').text(data.ipRgbYcbcrKlor + ' μg/g').show();

				$('#antosianin-ip-hsv').text(data.ipHsvAnto + ' abs/g').show();
				$('#karotenoid-ip-hsv').text(data.ipHsvKaro + ' μg/g').show();
				$('#klorofil-ip-hsv').text(data.ipHsvKlor + ' μg/g').show();

				$('#antosianin-ip-hsvlab').text(data.ipHsvLabAnto + ' abs/g').show();
				$('#karotenoid-ip-hsvlab').text(data.ipHsvLabKaro + ' μg/g').show();
				$('#klorofil-ip-hsvlab').text(data.ipHsvLabKlor + ' μg/g').show();

				$('#antosianin-ip-hsvycbcr').text(data.ipHsvYcbcrAnto + ' abs/g').show();
				$('#karotenoid-ip-hsvycbcr').text(data.ipHsvYcbcrKaro + ' μg/g').show();
				$('#klorofil-ip-hsvycbcr').text(data.ipHsvYcbcrKlor + ' μg/g').show();

				$('#antosianin-ip-lab').text(data.ipLabAnto + ' abs/g').show();
				$('#karotenoid-ip-lab').text(data.ipLabKaro + ' μg/g').show();
				$('#klorofil-ip-lab').text(data.ipLabKlor+ ' μg/g').show();

				$('#antosianin-ip-labycbcr').text(data.ipLabYcbcrAnto + ' abs/g').show();
				$('#karotenoid-ip-labycbcr').text(data.ipLabYcbcrKaro + ' μg/g').show();
				$('#klorofil-ip-labycbcr').text(data.ipLabYcbcrKlor + ' μg/g').show();
								
				$('#antosianin-ip-ycbcr').text(data.ipYCbCrAnto + ' abs/g').show();
				$('#karotenoid-ip-ycbcr').text(data.ipYCbCrKaro + ' μg/g').show();
				$('#klorofil-ip-ycbcr').text(data.ipYCbCrKlor + ' μg/g').show();
				
				$('#errorAlert').hide();
				
				$('#loading').hide();
				$('#process-wait').hide();
				console.log(data.error);
			}
			
		});

		
		event.preventDefault();

	});

});