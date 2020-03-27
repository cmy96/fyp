html_layout = '''<!DOCTYPE html>
                    <html>

                    <meta name="viewport"  content = "width=device-width, initial-scale=1.0, user-scalable=yes">
                    <script src="https://kit.fontawesome.com/a10c0e47f2.js" crossorigin="anonymous"></script>
                        <head>
                            {%metas%}
                            <title>{%title%}</title>
                            {%favicon%}
                            {%css%}

                        </head>

                        <body>

                            <div class="navbar">
                                <nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top py-3" id="mainNav">
                                    <a class="navbar-brand js-scroll-trigger" href="/"><img src="/assets/logo-1.png", width="35px", height="30px", align='center'>C.A.R.E</a>
                                    <span class="navbar-text navText">Breast Cancer Prediction Tool</span>

                                    <button class="navbar-toggler navbar-toggler-right" type="button" data-toggle="collapse" data-target="#navbarResponsive" aria-controls="navbarResponsive" aria-expanded="false" aria-label="Toggle navigation">
                                        <span class="navbar-toggler-icon"></span>
                                    </button>

                                    <div class="collapse navbar-collapse" id="navbarResponsive">
                                        <ul class="navbar-nav ml-auto">
                                            <li class="nav-item">
                                                <a class="nav-link js-scroll-trigger" id="prediction" href="/index2.html"><i class="fas fa-robot"></i> Prediction </a>
                                            </li>
                                            <li class="nav-item">
                                                <a class="nav-link js-scroll-trigger" id="dashboard" href="/dashboard/bills"><i class="fas fa-chart-pie"></i> Dashboard</a>
                                            </li>
                                        </ul>
                                    </div>
                                </nav>
                            </div>




                            <div class="app_entry">
                                {%app_entry%}
                            </div>




                            <footer>
                                {%config%}
                                {%scripts%}
                                {%renderer%}
                            </footer>

                            <script>
                            $('body').append('<div style="" id="loadingDiv"><div class="loader">Loading...</div></div>');
                                $(window).on('load', function(){
                                setTimeout(removeLoader, 2000); //wait for page load PLUS two seconds.
                                });
                                function removeLoader(){
                                    $( "#loadingDiv" ).fadeOut(500, function() {
                                    // fadeOut complete. Remove the loading div
                                    $( "#loadingDiv" ).hide(); //makes page more lightweight 
                                });  
                                };
                            
                            $(document).ready(function() {

                                var url = location.pathname;

                                    if (url.indexOf('dashboard') > -1) {
                                        $('#dashboard').addClass("active");
                                    } else if (url.indexOf('results') > -1) {
                                        $('#prediction').addClass("active");
                                    } else {
                                        $('#prediction').addClass("active");
                                    }

                                });     
                            </script>
                            <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js"></script>

                        </body>
                    </html>'''
