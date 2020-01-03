import React, { Component } from 'react';
import logo from './Logo.png';
import 'bootstrap/dist/css/bootstrap.min.css';
//import {Button, ButtonToolbar} from 'react-bootstrap';
import { Navbar, Nav, NavDropdown, Form, FormControl, Button, Spinner } from 'react-bootstrap';
import Container from 'react-bootstrap/Container'
import Col from 'react-bootstrap/Col'
import Row from 'react-bootstrap/Row'
import ButtonGroup from 'react-bootstrap/ButtonGroup'
import Jumbotron from 'react-bootstrap/Jumbotron'
import DropdownButton from 'react-bootstrap/DropdownButton'
import Dropdown from 'react-bootstrap/Dropdown'
//import Form from 'react-bootstrap/Form'
import './App.css';

function App() {
  return (

    <div className="App">

      <Navbar bg="light" expand="lg">
        <img
          src={logo}
          width="70"
          height="70"
          className="d-inline-block align-top"   
        />
        <Navbar.Brand href="#home">C.A.R.E</Navbar.Brand>
        Breast Cancer Prediction Tool
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="mr-auto">
            <Nav.Link href="#home">Home</Nav.Link>
            <Nav.Link href="#link">Link</Nav.Link>
            <NavDropdown title="Dropdown" id="basic-nav-dropdown">
              <NavDropdown.Item href="#action/3.1">Action</NavDropdown.Item>
              <NavDropdown.Item href="#action/3.2">Another action</NavDropdown.Item>
              <NavDropdown.Item href="#action/3.3">Something</NavDropdown.Item>
              <NavDropdown.Divider />
              <NavDropdown.Item href="#action/3.4">Separated link</NavDropdown.Item>
            </NavDropdown>
          </Nav>
        </Navbar.Collapse>
      </Navbar>

      <h2 id='calculator_header'>Calculator</h2>

      <Form>
        <Container id="container">
          <Row >
            <Col xs={6}>
              <Form.Group as={Row} controlId="formHorizontalAge">
                <Form.Label column sm={4}>
                  Age at Diagnosis
                </Form.Label>
                <Col sm={6}>
                  <Form.Control type="text" placeholder="Age" />
                </Col>
              </Form.Group>

              <Form.Group as={Row} controlId="formHorizontalHer2">
                <Form.Label column sm={4}>
                  Her2
                </Form.Label>
                <Col sm={6}>
                  <ButtonGroup aria-label="Basic example">
                    <Button variant="Positive">Positive</Button>
                    <Button variant="Negative">Negative</Button>
                  </ButtonGroup>
                </Col>
              </Form.Group>

              <Form.Group as={Row} controlId="formHorizontalHer2">
                <Form.Label column sm={4}>
                  ER
                </Form.Label>
                <Col sm={6}>
                  <ButtonGroup aria-label="Basic example">
                    <Button variant="Positive">Positive</Button>
                    <Button variant="Negative">Negative</Button>
                  </ButtonGroup>
                </Col>
              </Form.Group>

              <Form.Group as={Row} controlId="formHorizontalHer2">
                <Form.Label column sm={4}>
                  PR
                </Form.Label>
                <Col sm={6}>
                  <ButtonGroup aria-label="Basic example">
                    <Button variant="Positive" >Positive</Button>
                    <Button variant="Negative">Negative</Button>
                  </ButtonGroup>
                </Col>
              </Form.Group>

              <Form.Group as={Row} controlId="formHorizontalStage">
                <Form.Label column sm={4}>
                  Stage
                </Form.Label>
                <Col sm={6}>
                <Form.Control type="text" placeholder="Stage" />

                </Col>
              </Form.Group>

            </Col>

            <Col xs={{span: 4, offset: 1 }}>
              <Form.Group as={Row} controlId="formHorizontalTumorSize">
                <Form.Label column sm={3}>Tumor Size</Form.Label>
                <Col sm={4}>
                  <Form.Control as="select">
                    <option>Tx</option>
                    <option>Tis</option>
                    <option>T0</option>
                    <option>T1</option>
                    <option>T2</option>
                    <option>T3</option>
                    <option>T4</option>
                  </Form.Control>
                </Col>
              </Form.Group>

              <Form.Group as={Row} controlId="formHorizontalNode">
                <Form.Label column sm={3}>
                  Nodes
                </Form.Label>
                
                <Col sm={4}>
                <Form.Control as="select">
                <option>Nx</option>
                  <option>N0</option>
                  <option>N1</option>
                  <option>N2</option>
                  <option>N3</option>
                </Form.Control>
                </Col>
              </Form.Group>

              <Form.Group as={Row} controlId="formHorizontalMetastasis">
                <Form.Label column sm={3}>
                  Metastasis
                </Form.Label>
                
                <Col sm={4}>
                  <Form.Control as="select">
                    <option>Mx</option>
                    <option>M0</option>
                    <option>M1</option>
  
                </Form.Control>
                </Col>
              </Form.Group>

            </Col>

          </Row>
        </Container>
      <div id="submit">
      <Form.Group as={Row} id="submit">
              <Button type="submit">Calculate</Button>
          </Form.Group>
          </div>
    </Form>
      




  </div>
  );
}



export default App;



/*<Spinner animation="border" role="status">
        <span className="sr-only">Loading...</span>
      </Spinner>*/