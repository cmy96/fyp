html_layout = '''<!DOCTYPE html>
                    <html>
                    <script src="https://ajax.googleapis.com/ajax/libs/jquery/1.8.3/jquery.min.js"></script>
                    <script src="http://code.jquery.com/ui/1.9.2/jquery-ui.js"></script>
                    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.1/jquery.min.js"></script>
                    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
                    <script type="text/javascript" src="{{ url_for('static', filename='dist/js/main.js') }}"></script>
                            
                    <link rel="stylesheet" media="screen" href = "{{ url_for('static', filename='bootstrap.min.css') }}">
                    <link rel="stylesheet" media="screen" href = "{{ url_for('static', filename='dist/css/styles.css') }}">
                    <meta name="viewport"  content = "width=device-width, initial-scale=1.0, user-scalable=yes">
                    <script src="https://kit.fontawesome.com/a10c0e47f2.js" crossorigin="anonymous"></script>

                    <meta name="viewport"  content = "width=device-width, initial-scale=1.0, user-scalable=yes">
                    <script src="https://kit.fontawesome.com/a10c0e47f2.js" crossorigin="anonymous"></script>
                        <head>
                            {%metas%}
                            <title>{%title%}</title>
                            {%favicon%}
                            {%css%}

                        </head>

                        <body>



                              <nav class="navbar navbar-expand-lg navbar-dark bg-dark fixed-top py-3">
                                    <a class="navbar-brand" href="/"><img src="/assets/logo-1.png", width="35px", height="30px", align='center'>C.A.R.E</a>
                                    <span class="navbar-text navText" style="color:white;">Breast Cancer Prediction Tool</span>
                                    <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                                        <span class="navbar-toggler-icon"></span>
                                    </button>
                                    
                                    <div class="collapse navbar-collapse" id="navbarSupportedContent">
                                        <ul class="navbar-nav ml-auto nav">
                                        <li class="nav-item ">
                                            <a class="nav-link" href="/index2.html"><i class="fas fa-calculator"></i> Calculator <span class="sr-only">(current)</span></a>
                                        </li>
                                        <li class="nav-item active">
                                            <a class="nav-link" href="/dashboard/bills"><i class="fas fa-chart-pie"></i> Dashboard</a>
                                        </li>
                                
                                        </ul>

                                    </div>
                                </nav>




                            <div class="app_entry">
                                {%app_entry%}
                            </div>


                            <footer>
                                {%config%}
                                {%scripts%}
                                {%renderer%}
                            </footer>
                        </body>
                    </html>'''
