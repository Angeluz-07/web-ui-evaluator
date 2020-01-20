

var app = new Vue({        
    delimiters: ['[[', ']]'],  
    el: '#app',
    data : {
        predictionURL : '/api/predict/',
        currentImage : null, //to display        
        currentImageFile : null, //to send in POST
        currentPrediction : null
    },
    methods : {
        updatePrediction : function(){
            var data = new FormData();
            data.append('name', 'web_picture');
            data.append('file', this.currentImageFile );
            var config = {
                header : {
                    'Content-Type' : 'image/png'
                }
            };
            axios.post(this.predictionURL, data, config)
            .then(function (response) {               
                // handle success
                this.currentPrediction = response.data;

                chart.data.datasets[0].data[0] = parseFloat(this.currentPrediction.appealing_contrast).toFixed(5);
                chart.data.datasets[0].data[1] = parseFloat(this.currentPrediction.minimalist).toFixed(5);
                chart.data.datasets[0].data[2] = parseFloat(this.currentPrediction.visual_load).toFixed(5);
                chart.update();

                console.log(this.currentPrediction);
            }).catch((err) => {
                this.loading = false;
                console.log(err);
            });

        },        
        onUploadImage : function(e){
            // https://stackoverflow.com/questions/45071661/how-can-i-display-image-by-image-uploaded-on-the-vue-component
            var file = e.target.files[0];
            this.currentImageFile = file;

            var reader = new FileReader();
            reader.onload = (e) => {                
                this.currentImage = e.target.result;
            };
            reader.readAsDataURL(file);
        },
    }
});


window.chartColors = {
	red: 'rgb(255, 99, 132)',
	orange: 'rgb(255, 159, 64)',
	yellow: 'rgb(255, 205, 86)',
	green: 'rgb(75, 192, 192)',
	blue: 'rgb(54, 162, 235)',
	purple: 'rgb(153, 102, 255)',
	grey: 'rgb(201, 203, 207)'
};

var ctx = document.getElementById('results').getContext('2d');
var chart = new Chart(ctx, {
    // The type of chart we want to create
    type: 'doughnut',

    // The data for our dataset
    data: {
        datasets: [{
            data: [null, null, null],
            backgroundColor : [
                chartColors.yellow,
                chartColors.grey,
                chartColors.blue
            ]
        }],
    
        // These labels appear in the legend and in the tooltips when hovering different arcs
        labels: [
            'Appealing Contrast Prob.',
            'Minimalist Prob.',
            'Visual Load Prob.'
        ]
    },

    // Configuration options go here
    options: {
        rotation : 1 * Math.PI
    }
});

// Add the following code if you want the name of the file appear on select
$(".custom-file-input").on("change", function() {
    var fileName = $(this).val().split("\\").pop();
    $(this).siblings(".custom-file-label").addClass("selected").html(fileName);
  });